//! Reinforcement learning with three factors: a presynaptic spike leaves a synaptic eligibility
//! trace, a postsynaptic response gates it, and a broadcast reward-prediction error turns it into
//! a weight change — temporal-difference learning done the way a chip with a global neuromodulator
//! does it, checked against the Bellman solution of the task.
//!
//! # What the mechanism is
//!
//! Temporal-difference learning (Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd
//! ed., MIT Press, 2018, ch. 6 and 12) keeps a value estimate `V(s)` and, on every transition
//! `s → s'` with reward `r`, computes the prediction error `δ = r + γ V(s') − V(s)`. With
//! **eligibility traces** the error is credited not only to the state just left but to every state
//! visited recently, each weighted by a trace that decays by `γλ` per step — TD(λ). The
//! three-factor reading (Frémaux, Sprekeler and Gerstner, *Reinforcement learning using a
//! continuous time actor-critic framework with spiking neurons*, `PLoS` Computational Biology
//! 9(4):e1003024, 2013; Izhikevich, *Solving the distal reward problem through linkage of STDP and
//! dopamine signaling*, Cerebral Cortex 17(10):2443–2452, 2007) puts the trace **in the synapse**:
//! a presynaptic spike from the cell that represents the state deposits eligibility, the trace
//! decays on its own, and the weight moves only when the error arrives, by `η δ e`. Nothing has to
//! remember the trajectory; the synapses do.
//!
//! # Why it is in a neuromorphic crate
//!
//! This is the learning rule neuromorphic hardware implements natively — a per-synapse trace and
//! a broadcast third factor are what Loihi's programmable learning engine and `SpiNNaker`'s
//! plasticity kernels provide — and it is the rule whose cost the trace decides: a weight update
//! touches only the synapses whose trace is above the floor: exactly one at `λ = 0`, and at most
//! `ln(floor) / ln(γλ)` — 43 at `γλ = 0.72` with a floor of `1e-6` — however long the trajectory,
//! while the *weight* of the credit those synapses share sums to `1/(1 − γλ)`. [`Critic::update`]
//! reports how many it touched, so the saving is a count and not an argument.
//!
//! # The closed forms this module is checked against
//!
//! - Value iteration on a chain of `n` states with a unit reward on the last step and discount `γ`
//!   gives `V(s) = γ^{n − 2 − s}` (the reward arrives on the `(n − 2 − s)`-th step after leaving
//!   `s`, discounted that many times) — the oracle is checked against that literal before it
//!   referees. The first draft wrote `n − 1 − s`, one step too many, and the oracle caught it.
//! - The eligibility trace after one deposit and `k` idle steps is `(γλ)^k`, exactly.
//! - TD(0) and TD(0.8) both converge to the oracle's values on the chain, and after a fixed short
//!   budget of episodes TD(λ) is closer than TD(0) — the credit-assignment speed-up, measured.
//! - A softmax actor with the same trace learns to walk the chain: the probability of the correct
//!   action exceeds 0.9 at every state.
//! - At `λ = 0` every update touches exactly one synapse; at `λ = 0.8, γ = 0.9` it touches at most
//!   `⌈ln(1e-6)/ln(0.72)⌉ + 1 = 43`, and on a 60-state chain far fewer than the states on average.
//!
//! # What this module has NOT reproduced
//!
//! - The continuous-time spiking actor-critic of Frémaux et al. with its place cells and LIF
//!   populations. The state code here is tabular — one cell per state, one spike per step while
//!   occupied — which is the limit in which their rule is TD(λ) exactly, and the point of
//!   this module is that limit.
//! - Any benchmark task beyond a chain. The chain has a closed form; a maze has a lookup table.

use core::fmt;

use crate::rng::Rng;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ReinforceError {
    /// A count of zero where at least one is needed.
    Empty {
        /// What was empty.
        what: &'static str,
    },
    /// A state or action index past the task.
    Index {
        /// Which index.
        what: &'static str,
        /// The value.
        index: usize,
        /// The count it had to be below.
        count: usize,
    },
    /// A parameter outside its range.
    OutOfRange {
        /// Which parameter.
        what: &'static str,
        /// Value supplied.
        value: f64,
        /// Lowest admissible.
        low: f64,
        /// Highest admissible.
        high: f64,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Position in the offending array, `0` for a scalar.
        index: usize,
    },
}

impl fmt::Display for ReinforceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::Index { what, index, count } => write!(f, "{what} {index} is past the {count} available"),
            Self::OutOfRange { what, value, low, high } => {
                write!(f, "{what} = {value} is outside [{low}, {high}]")
            }
            Self::NonFinite { what, index } => write!(f, "{what} is not finite at {index}"),
        }
    }
}

impl std::error::Error for ReinforceError {}

fn unit_interval(what: &'static str, v: f64, high_open: bool) -> Result<f64, ReinforceError> {
    let ok = v.is_finite() && v >= 0.0 && if high_open { v < 1.0 } else { v <= 1.0 };
    if ok {
        Ok(v)
    } else {
        Err(ReinforceError::OutOfRange { what, value: v, low: 0.0, high: 1.0 })
    }
}

fn positive(what: &'static str, v: f64) -> Result<f64, ReinforceError> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(ReinforceError::OutOfRange { what, value: v, low: f64::MIN_POSITIVE, high: f64::INFINITY })
    }
}

// ---------------------------------------------------------------------------------------------
// The task
// ---------------------------------------------------------------------------------------------

/// A deterministic finite task: `next[s][a]` and `reward[s][a]`, with terminal states.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// States.
    pub states: usize,
    /// Actions.
    pub actions: usize,
    /// `next[s * actions + a]`.
    pub next: Vec<usize>,
    /// `reward[s * actions + a]`.
    pub reward: Vec<f64>,
    /// Whether a state is terminal: entering it ends the episode and its value is zero.
    pub terminal: Vec<bool>,
    /// Discount `γ` in `[0, 1)`.
    pub gamma: f64,
}

impl Task {
    /// Build.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Empty`] for no states or actions, [`ReinforceError::Index`] for a
    /// transition past the states, [`ReinforceError::OutOfRange`] for a `gamma` outside `[0, 1)`,
    /// [`ReinforceError::NonFinite`] for a non-finite reward, and a length mismatch as
    /// [`ReinforceError::Index`] naming the array.
    pub fn new(
        states: usize,
        actions: usize,
        next: Vec<usize>,
        reward: Vec<f64>,
        terminal: Vec<bool>,
        gamma: f64,
    ) -> Result<Self, ReinforceError> {
        if states == 0 {
            return Err(ReinforceError::Empty { what: "states" });
        }
        if actions == 0 {
            return Err(ReinforceError::Empty { what: "actions" });
        }
        let want = states * actions;
        if next.len() != want {
            return Err(ReinforceError::Index { what: "next (length)", index: next.len(), count: want });
        }
        if reward.len() != want {
            return Err(ReinforceError::Index { what: "reward (length)", index: reward.len(), count: want });
        }
        if terminal.len() != states {
            return Err(ReinforceError::Index { what: "terminal (length)", index: terminal.len(), count: states });
        }
        if let Some(&n) = next.iter().find(|n| **n >= states) {
            return Err(ReinforceError::Index { what: "next state", index: n, count: states });
        }
        if let Some(i) = reward.iter().position(|r| !r.is_finite()) {
            return Err(ReinforceError::NonFinite { what: "reward", index: i });
        }
        unit_interval("gamma", gamma, true)?;
        Ok(Self { states, actions, next, reward, terminal, gamma })
    }

    /// A chain of `n` states: action `1` moves right, action `0` moves left (or stays at the
    /// start); the last state is terminal and entering it pays `1`. Discount `gamma`.
    ///
    /// Under the always-right policy `V(s) = γ^{n − 2 − s}` for every non-terminal `s`: the unit
    /// reward is paid on the step into the terminal state, `n − 2 − s` steps after leaving `s`.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::OutOfRange`] for `n < 2` or a bad `gamma`.
    pub fn chain(n: usize, gamma: f64) -> Result<Self, ReinforceError> {
        if n < 2 {
            return Err(ReinforceError::OutOfRange { what: "n", value: n as f64, low: 2.0, high: f64::INFINITY });
        }
        let mut next = vec![0; n * 2];
        let mut reward = vec![0.0; n * 2];
        let mut terminal = vec![false; n];
        terminal[n - 1] = true;
        for s in 0..n {
            next[s * 2] = s.saturating_sub(1);
            next[s * 2 + 1] = (s + 1).min(n - 1);
            if s + 1 == n - 1 {
                reward[s * 2 + 1] = 1.0;
            }
        }
        // The terminal state maps to itself with no reward.
        next[(n - 1) * 2] = n - 1;
        next[(n - 1) * 2 + 1] = n - 1;
        reward[(n - 1) * 2 + 1] = 0.0;
        Self::new(n, 2, next, reward, terminal, gamma)
    }

    /// The exact state values of a fixed policy `policy[s] ∈ actions`, by value iteration to
    /// `tol`: the oracle every learner here is checked against.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Index`] for a policy of the wrong length or an action past `actions`,
    /// [`ReinforceError::OutOfRange`] for a non-positive `tol`.
    pub fn values_of(&self, policy: &[usize], tol: f64) -> Result<Vec<f64>, ReinforceError> {
        if policy.len() != self.states {
            return Err(ReinforceError::Index { what: "policy (length)", index: policy.len(), count: self.states });
        }
        if let Some(&a) = policy.iter().find(|a| **a >= self.actions) {
            return Err(ReinforceError::Index { what: "action", index: a, count: self.actions });
        }
        positive("tol", tol)?;
        let mut v = vec![0.0; self.states];
        for _ in 0..1_000_000 {
            let mut worst = 0.0f64;
            for s in 0..self.states {
                if self.terminal[s] {
                    continue;
                }
                let a = policy[s];
                let n = self.next[s * self.actions + a];
                let target = self.reward[s * self.actions + a] + self.gamma * v[n];
                worst = worst.max((target - v[s]).abs());
                v[s] = target;
            }
            if worst < tol {
                break;
            }
        }
        Ok(v)
    }
}

// ---------------------------------------------------------------------------------------------
// Eligibility and the critic
// ---------------------------------------------------------------------------------------------

/// The synaptic eligibility trace: `e ← γλ e` every step, `+1` on a presynaptic spike.
#[derive(Debug, Clone, PartialEq)]
pub struct Eligibility {
    /// One trace per synapse.
    pub e: Vec<f64>,
    /// `γλ`, the per-step decay.
    pub decay: f64,
}

impl Eligibility {
    /// `n` traces at zero with the given decay.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Empty`] for `n = 0`, [`ReinforceError::OutOfRange`] for a decay outside
    /// `[0, 1)`.
    pub fn new(n: usize, decay: f64) -> Result<Self, ReinforceError> {
        if n == 0 {
            return Err(ReinforceError::Empty { what: "synapses" });
        }
        unit_interval("decay", decay, true)?;
        Ok(Self { e: vec![0.0; n], decay })
    }

    /// Decay every trace by one step.
    pub fn step(&mut self) {
        for x in &mut self.e {
            *x *= self.decay;
        }
    }

    /// A presynaptic spike on synapse `i` (accumulating: `+1`).
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Index`].
    pub fn spike(&mut self, i: usize) -> Result<(), ReinforceError> {
        if i >= self.e.len() {
            return Err(ReinforceError::Index { what: "synapse", index: i, count: self.e.len() });
        }
        self.e[i] += 1.0;
        Ok(())
    }

    /// How many traces are above `floor` — the synapses an update has to touch.
    #[must_use]
    pub fn active(&self, floor: f64) -> usize {
        self.e.iter().filter(|x| **x > floor).count()
    }

    /// Zero every trace.
    pub fn reset(&mut self) {
        self.e.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// A tabular critic — one synapse per state onto a value cell — learned by TD(λ) with a synaptic
/// eligibility trace and the prediction error as the third factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Critic {
    /// Value estimates, one per state.
    pub v: Vec<f64>,
    /// The traces.
    pub trace: Eligibility,
    /// Learning rate.
    pub eta: f64,
    /// Discount `γ`.
    pub gamma: f64,
    /// Traces below this are not visited by an update. `1e-6`.
    pub floor: f64,
    /// Synapses touched by updates so far.
    pub touched: u64,
    /// Updates performed.
    pub updates: u64,
}

impl Critic {
    /// A critic over `states` with TD(λ) at discount `gamma`.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Empty`], [`ReinforceError::OutOfRange`] for `eta ≤ 0`, `gamma` or
    /// `lambda` outside `[0, 1)` / `[0, 1]`.
    pub fn new(states: usize, eta: f64, gamma: f64, lambda: f64) -> Result<Self, ReinforceError> {
        if states == 0 {
            return Err(ReinforceError::Empty { what: "states" });
        }
        positive("eta", eta)?;
        unit_interval("gamma", gamma, true)?;
        unit_interval("lambda", lambda, false)?;
        Ok(Self {
            v: vec![0.0; states],
            trace: Eligibility::new(states, gamma * lambda)?,
            eta,
            gamma,
            floor: 1e-6,
            touched: 0,
            updates: 0,
        })
    }

    /// One transition: the state cell of `s` spikes into its trace, the error
    /// `δ = r + γ V(s') − V(s)` (with `V(s') = 0` for a terminal `s'`) is broadcast, and every
    /// synapse whose trace is above the floor moves by `η δ e`. Returns `δ`.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Index`] for a state past the table, [`ReinforceError::NonFinite`] for a
    /// non-finite reward.
    pub fn update(&mut self, s: usize, r: f64, s_next: usize, next_terminal: bool) -> Result<f64, ReinforceError> {
        let n = self.v.len();
        if s >= n {
            return Err(ReinforceError::Index { what: "state", index: s, count: n });
        }
        if s_next >= n {
            return Err(ReinforceError::Index { what: "next state", index: s_next, count: n });
        }
        if !r.is_finite() {
            return Err(ReinforceError::NonFinite { what: "reward", index: 0 });
        }
        self.trace.step();
        self.trace.spike(s)?;
        let v_next = if next_terminal { 0.0 } else { self.v[s_next] };
        let delta = r + self.gamma * v_next - self.v[s];
        let mut touched = 0u64;
        for (w, &e) in self.v.iter_mut().zip(&self.trace.e) {
            if e > self.floor {
                *w += self.eta * delta * e;
                touched += 1;
            }
        }
        self.touched += touched;
        self.updates += 1;
        Ok(delta)
    }

    /// Start a new episode: traces to zero, values kept.
    pub fn end_episode(&mut self) {
        self.trace.reset();
    }

    /// Synapses touched per update so far. `None` before any update.
    #[must_use]
    pub fn touched_per_update(&self) -> Option<f64> {
        if self.updates == 0 { None } else { Some(self.touched as f64 / self.updates as f64) }
    }
}

// ---------------------------------------------------------------------------------------------
// The actor
// ---------------------------------------------------------------------------------------------

/// A tabular softmax actor learned by the same three factors: the eligibility of the taken
/// action's synapse is the policy-gradient term `1 − π(a|s)` (and `−π(a'|s)` for the others),
/// decaying by `γλ`, and the critic's `δ` is the third factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Actor {
    /// Preferences `θ[s * actions + a]`.
    pub theta: Vec<f64>,
    /// Actions.
    pub actions: usize,
    /// The traces, one per `(state, action)`.
    pub trace: Eligibility,
    /// Learning rate.
    pub eta: f64,
}

impl Actor {
    /// An actor over `states × actions` at decay `γλ`.
    ///
    /// # Errors
    ///
    /// As [`Eligibility::new`], plus [`ReinforceError::OutOfRange`] for `eta ≤ 0`.
    pub fn new(states: usize, actions: usize, eta: f64, decay: f64) -> Result<Self, ReinforceError> {
        positive("eta", eta)?;
        if actions == 0 {
            return Err(ReinforceError::Empty { what: "actions" });
        }
        if states == 0 {
            return Err(ReinforceError::Empty { what: "states" });
        }
        Ok(Self { theta: vec![0.0; states * actions], actions, trace: Eligibility::new(states * actions, decay)?, eta })
    }

    /// The policy `π(·|s)`.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Index`].
    pub fn policy(&self, s: usize) -> Result<Vec<f64>, ReinforceError> {
        let states = self.theta.len() / self.actions;
        if s >= states {
            return Err(ReinforceError::Index { what: "state", index: s, count: states });
        }
        let row = &self.theta[s * self.actions..(s + 1) * self.actions];
        let top = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|t| (t - top).exp()).collect();
        let z: f64 = exps.iter().sum();
        Ok(exps.iter().map(|e| e / z).collect())
    }

    /// Sample an action in `s` and deposit the policy-gradient eligibility.
    ///
    /// # Errors
    ///
    /// [`ReinforceError::Index`].
    pub fn act(&mut self, s: usize, rng: &mut Rng) -> Result<usize, ReinforceError> {
        let p = self.policy(s)?;
        let u = rng.next_f64();
        let mut acc = 0.0;
        let mut a = self.actions - 1;
        for (k, &pk) in p.iter().enumerate() {
            acc += pk;
            if u < acc {
                a = k;
                break;
            }
        }
        self.trace.step();
        for (k, &pk) in p.iter().enumerate() {
            let g = if k == a { 1.0 - pk } else { -pk };
            self.trace.e[s * self.actions + k] += g;
        }
        Ok(a)
    }

    /// Apply the third factor: `θ += η δ e` on every synapse.
    pub fn reinforce(&mut self, delta: f64) {
        for (t, &e) in self.theta.iter_mut().zip(&self.trace.e) {
            *t += self.eta * delta * e;
        }
    }

    /// Start a new episode.
    pub fn end_episode(&mut self) {
        self.trace.reset();
    }
}

/// Run one episode of at most `max_steps` from `start`, with the actor choosing and the critic
/// learning; returns the return `Σ γ^t r_t` and the steps taken.
///
/// # Errors
///
/// [`ReinforceError::Index`] for a start past the task, and anything the pieces refuse.
pub fn episode(
    task: &Task,
    critic: &mut Critic,
    actor: &mut Actor,
    start: usize,
    max_steps: usize,
    rng: &mut Rng,
) -> Result<(f64, usize), ReinforceError> {
    if start >= task.states {
        return Err(ReinforceError::Index { what: "start", index: start, count: task.states });
    }
    let mut s = start;
    let mut ret = 0.0;
    let mut disc = 1.0;
    let mut steps = 0;
    while steps < max_steps && !task.terminal[s] {
        let a = actor.act(s, rng)?;
        let n = task.next[s * task.actions + a];
        let r = task.reward[s * task.actions + a];
        let delta = critic.update(s, r, n, task.terminal[n])?;
        actor.reinforce(delta);
        ret += disc * r;
        disc *= task.gamma;
        s = n;
        steps += 1;
    }
    critic.end_episode();
    actor.end_episode();
    Ok((ret, steps))
}

#[cfg(test)]
mod tests {
    use super::{Actor, Critic, Eligibility, ReinforceError, Task, episode};
    use crate::rng::Rng;

    /// The oracle against its own closed form: on the chain, `V(s) = γ^{n−1−s}` to 1e-12.
    #[test]
    fn value_iteration_on_the_chain_is_the_geometric_closed_form() {
        let n = 8;
        let gamma = 0.9;
        let task = Task::chain(n, gamma).unwrap();
        let right = vec![1usize; n];
        let v = task.values_of(&right, 1e-14).unwrap();
        for s in 0..n - 1 {
            let want = gamma.powi((n - 2 - s) as i32);
            assert!((v[s] - want).abs() < 1e-12, "V({s}) = {} vs {want}", v[s]);
        }
        assert_eq!(v[n - 2], 1.0, "one step from the reward, undiscounted");
        assert_eq!(v[n - 1], 0.0, "the terminal state is worth nothing");
        // The always-left policy never reaches the reward: every value is zero.
        let left = vec![0usize; n];
        assert!(task.values_of(&left, 1e-14).unwrap().iter().all(|x| *x == 0.0));
    }

    /// One deposit, `k` idle steps: `(γλ)^k` exactly.
    #[test]
    fn the_eligibility_trace_decays_as_gamma_lambda_to_the_k() {
        let mut e = Eligibility::new(3, 0.9 * 0.8).unwrap();
        e.spike(1).unwrap();
        assert_eq!(e.e, vec![0.0, 1.0, 0.0]);
        for _ in 0..5 {
            e.step();
        }
        assert!((e.e[1] - 0.72f64.powi(5)).abs() < 1e-15);
        assert_eq!(e.active(1e-6), 1);
        e.spike(1).unwrap();
        assert!((e.e[1] - (0.72f64.powi(5) + 1.0)).abs() < 1e-15, "accumulating, not replacing");
        e.reset();
        assert_eq!(e.active(0.0), 0);
        assert!(matches!(e.spike(3), Err(ReinforceError::Index { .. })));
        assert!(matches!(Eligibility::new(0, 0.5), Err(ReinforceError::Empty { .. })));
        assert!(matches!(Eligibility::new(1, 1.0), Err(ReinforceError::OutOfRange { .. })));
    }

    /// TD(0) and TD(0.8) both converge to the oracle on the chain under the always-right policy;
    /// after a short budget TD(λ) is the closer — credit assignment travels the whole trajectory
    /// in one episode with a trace and one state per episode without.
    #[test]
    fn td_converges_to_the_oracle_and_the_trace_gets_there_faster() {
        let n = 10;
        let gamma = 0.9;
        let task = Task::chain(n, gamma).unwrap();
        let oracle = task.values_of(&vec![1usize; n], 1e-14).unwrap();
        let run = |lambda: f64, episodes: usize| -> Vec<f64> {
            let mut c = Critic::new(n, 0.2, gamma, lambda).unwrap();
            for _ in 0..episodes {
                let mut s = 0;
                while !task.terminal[s] {
                    let next = task.next[s * 2 + 1];
                    let r = task.reward[s * 2 + 1];
                    c.update(s, r, next, task.terminal[next]).unwrap();
                    s = next;
                }
                c.end_episode();
            }
            c.v.clone()
        };
        let err = |v: &[f64]| oracle.iter().zip(v).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let long0 = run(0.0, 400);
        let long8 = run(0.8, 400);
        assert!(err(&long0) < 0.02, "TD(0) after 400 episodes: worst error {}", err(&long0));
        assert!(err(&long8) < 0.02, "TD(0.8) after 400 episodes: worst error {}", err(&long8));
        let short0 = run(0.0, 6);
        let short8 = run(0.8, 6);
        assert!(err(&short8) < err(&short0), "after 6 episodes TD(0.8) at {} was not closer than TD(0) at {}", err(&short8), err(&short0));
        // TD(0) after six episodes has reached back only six states from the reward; the start
        // state's value is still exactly zero.
        assert_eq!(short0[0], 0.0);
        assert!(short8[0] > 0.0, "the trace carried credit to the start in six episodes");
    }

    /// The cost of an update is the traces above the floor: exactly one at λ = 0, at most the
    /// trajectory length at λ = 0.8 — and on a long chain far fewer than the states.
    #[test]
    fn an_update_touches_only_the_eligible_synapses() {
        let n = 60;
        let task = Task::chain(n, 0.9).unwrap();
        // Above a floor of 1e-6 a trace survives ln(1e-6)/ln(0.72) = 42.9 steps: 43 synapses at
        // most, whatever the trajectory length.
        let floor_steps = (1e-6f64.ln() / 0.72f64.ln()).ceil() + 1.0;
        for (lambda, bound) in [(0.0f64, 1.0f64), (0.8, floor_steps)] {
            let mut c = Critic::new(n, 0.1, 0.9, lambda).unwrap();
            let mut s = 0;
            let mut worst = 0.0f64;
            let mut before = 0;
            while !task.terminal[s] {
                let next = task.next[s * 2 + 1];
                c.update(s, task.reward[s * 2 + 1], next, task.terminal[next]).unwrap();
                worst = worst.max((c.touched - before) as f64);
                before = c.touched;
                s = next;
            }
            assert!(worst <= bound, "λ {lambda}: an update touched {worst} synapses");
            if lambda == 0.0 {
                assert_eq!(c.touched_per_update(), Some(1.0));
            } else {
                // The count grows with the trajectory until it saturates at 43, so the average
                // over 59 steps is (43·44/2 + 16·43)/59 ≈ 28 — and `worst` reached the bound.
                assert!(worst >= 40.0, "the bound was never approached: {worst}");
                assert!(c.touched_per_update().unwrap() < 35.0);
                assert!(c.touched_per_update().unwrap() > 20.0);
            }
        }
        assert_eq!(Critic::new(3, 0.1, 0.9, 0.5).unwrap().touched_per_update(), None);
    }

    /// The actor learns to walk the chain: after training, the probability of stepping right
    /// exceeds 0.9 in every non-terminal state, the return approaches the oracle's `V(0)`, and
    /// before training it does not.
    #[test]
    fn the_actor_learns_to_walk_the_chain() {
        let n = 6;
        let gamma = 0.9;
        let task = Task::chain(n, gamma).unwrap();
        let mut rng = Rng::new(7);
        let mut critic = Critic::new(n, 0.1, gamma, 0.8).unwrap();
        let mut actor = Actor::new(n, 2, 0.5, gamma * 0.8).unwrap();
        let p0 = actor.policy(0).unwrap();
        assert!((p0[1] - 0.5).abs() < 1e-15, "an untrained actor is indifferent");
        let mut returns = Vec::new();
        for _ in 0..600 {
            let (ret, _) = episode(&task, &mut critic, &mut actor, 0, 200, &mut rng).unwrap();
            returns.push(ret);
        }
        for s in 0..n - 1 {
            let p = actor.policy(s).unwrap();
            assert!(p[1] > 0.9, "state {s}: P(right) = {}", p[1]);
        }
        let late: f64 = returns[500..].iter().sum::<f64>() / 100.0;
        let oracle = task.values_of(&vec![1usize; n], 1e-14).unwrap();
        assert!((late - oracle[0]).abs() < 0.1, "late return {late} against V(0) = {}", oracle[0]);
        let early: f64 = returns[..20].iter().sum::<f64>() / 20.0;
        assert!(early < late, "no learning: early {early}, late {late}");
        // The critic learned the values along the way.
        for s in 0..n - 1 {
            assert!((critic.v[s] - oracle[s]).abs() < 0.15, "V({s}) = {} vs {}", critic.v[s], oracle[s]);
        }
    }

    /// Every refusal names the problem.
    #[test]
    fn the_refusals_name_the_problem() {
        assert!(matches!(Task::chain(1, 0.9), Err(ReinforceError::OutOfRange { what: "n", .. })));
        assert!(matches!(Task::chain(3, 1.0), Err(ReinforceError::OutOfRange { what: "gamma", .. })));
        assert!(matches!(Task::new(0, 1, vec![], vec![], vec![], 0.5), Err(ReinforceError::Empty { what: "states" })));
        assert!(matches!(Task::new(2, 1, vec![0, 5], vec![0.0, 0.0], vec![false, true], 0.5), Err(ReinforceError::Index { what: "next state", index: 5, .. })));
        assert!(matches!(Task::new(2, 1, vec![0, 1], vec![0.0, f64::NAN], vec![false, true], 0.5), Err(ReinforceError::NonFinite { .. })));
        assert!(matches!(Task::new(2, 1, vec![0], vec![0.0, 0.0], vec![false, true], 0.5), Err(ReinforceError::Index { what: "next (length)", .. })));
        let task = Task::chain(3, 0.9).unwrap();
        assert!(matches!(task.values_of(&[1, 1], 1e-9), Err(ReinforceError::Index { what: "policy (length)", .. })));
        assert!(matches!(task.values_of(&[1, 2, 1], 1e-9), Err(ReinforceError::Index { what: "action", .. })));
        assert!(matches!(task.values_of(&[1, 1, 1], 0.0), Err(ReinforceError::OutOfRange { what: "tol", .. })));
        assert!(matches!(Critic::new(3, 0.0, 0.9, 0.5), Err(ReinforceError::OutOfRange { what: "eta", .. })));
        assert!(matches!(Critic::new(3, 0.1, 0.9, 1.5), Err(ReinforceError::OutOfRange { what: "lambda", .. })));
        let mut c = Critic::new(3, 0.1, 0.9, 0.5).unwrap();
        assert!(matches!(c.update(3, 0.0, 0, false), Err(ReinforceError::Index { what: "state", .. })));
        assert!(matches!(c.update(0, f64::INFINITY, 1, false), Err(ReinforceError::NonFinite { .. })));
        assert!(matches!(Actor::new(3, 0, 0.1, 0.5), Err(ReinforceError::Empty { what: "actions" })));
        let a = Actor::new(3, 2, 0.1, 0.5).unwrap();
        assert!(matches!(a.policy(3), Err(ReinforceError::Index { .. })));
        // The deposited eligibility is the policy gradient of ln π: `1 − π(a)` on the chosen
        // action and `−π(k)` on the others, which SUMS TO ZERO over the actions. With two actions
        // at 0.5 each, `π(a)` and `1 − π(a)` are the same number and the wrong one survived the
        // first mutation sweep; at π = (0.8, 0.2) they are not.
        let mut skew = Actor::new(1, 2, 0.1, 0.5).unwrap();
        skew.theta = vec![4.0f64.ln(), 0.0];
        let p = skew.policy(0).unwrap();
        assert!((p[0] - 0.8).abs() < 1e-12 && (p[1] - 0.2).abs() < 1e-12);
        let chosen = skew.act(0, &mut Rng::new(3)).unwrap();
        assert!((skew.trace.e[chosen] - (1.0 - p[chosen])).abs() < 1e-15, "chosen action's eligibility");
        assert!((skew.trace.e[0] + skew.trace.e[1]).abs() < 1e-15, "the eligibilities do not sum to zero: {:?}", skew.trace.e);
        // A terminal successor is worth zero whatever the table holds: the value field is public,
        // and an update into a terminal state must not read it.
        let mut c = Critic::new(3, 0.5, 0.9, 0.0).unwrap();
        c.v[2] = 5.0;
        let delta = c.update(1, 1.0, 2, true).unwrap();
        assert_eq!(delta, 1.0, "δ = r + γ·0 − V(1)");
        assert_eq!(c.v[1], 0.5);
        let mut rng = Rng::new(1);
        let mut c = Critic::new(3, 0.1, 0.9, 0.5).unwrap();
        let mut a = Actor::new(3, 2, 0.1, 0.5).unwrap();
        assert!(matches!(episode(&task, &mut c, &mut a, 9, 10, &mut rng), Err(ReinforceError::Index { what: "start", .. })));
        for e in [
            ReinforceError::Empty { what: "x" },
            ReinforceError::Index { what: "v", index: 3, count: 2 },
            ReinforceError::OutOfRange { what: "w", value: 9.0, low: 0.0, high: 1.0 },
            ReinforceError::NonFinite { what: "z", index: 0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }


    /// A transition INTO state `states` is one past the end; `>` for `≥` in that guard survived
    /// the second mutation sweep because every refusal tested was far past it.
    #[test]
    fn a_transition_to_one_past_the_last_state_is_refused() {
        let task = Task::new(2, 1, vec![2, 0], vec![0.0, 0.0], vec![false, false], 0.9);
        assert_eq!(task.unwrap_err(), ReinforceError::Index { what: "next state", index: 2, count: 2 });
        assert!(Task::new(2, 1, vec![1, 0], vec![0.0, 0.0], vec![false, false], 0.9).is_ok());
    }

    /// A parameter that must be positive refuses an infinity, and the refusal names the smallest
    /// admissible value rather than the excluded zero. The suite tested only `eta = 0`, which the
    /// comparison `v > 0.0` refuses on its own, and it matched every refusal with `..`, so the
    /// finiteness half of the guard and the `low` field it reports were both unread.
    #[test]
    fn a_positive_parameter_refuses_an_infinity_and_names_the_smallest_admissible_value() {
        let inf = f64::INFINITY;
        assert!(matches!(Critic::new(3, inf, 0.9, 0.5), Err(ReinforceError::OutOfRange { what: "eta", .. })));
        assert!(matches!(Actor::new(3, 2, inf, 0.5), Err(ReinforceError::OutOfRange { what: "eta", .. })));
        let task = Task::chain(3, 0.9).unwrap();
        assert!(matches!(task.values_of(&[1, 1, 1], inf), Err(ReinforceError::OutOfRange { what: "tol", .. })));
        // The bound the refusal names is the one that is admissible: `MIN_POSITIVE` is accepted and
        // zero is not, so reporting `low = 0.0` would name an inadmissible value as the floor.
        assert_eq!(
            Critic::new(3, 0.0, 0.9, 0.5).unwrap_err(),
            ReinforceError::OutOfRange { what: "eta", value: 0.0, low: f64::MIN_POSITIVE, high: inf }
        );
        assert!(Critic::new(3, f64::MIN_POSITIVE, 0.9, 0.5).is_ok());
    }

    /// Every refusal prints its own fields where its message says they go: the offending index
    /// before the count it had to be below. The suite asserted only that the four strings are
    /// non-empty, which any transposition of the fields satisfies.
    #[test]
    fn the_refusal_strings_print_their_fields_in_the_right_places() {
        assert_eq!(ReinforceError::Empty { what: "states" }.to_string(), "states is empty");
        assert_eq!(
            ReinforceError::Index { what: "next state", index: 5, count: 2 }.to_string(),
            "next state 5 is past the 2 available"
        );
        assert_eq!(
            ReinforceError::OutOfRange { what: "gamma", value: 9.0, low: 0.0, high: 1.0 }.to_string(),
            "gamma = 9 is outside [0, 1]"
        );
        assert_eq!(ReinforceError::NonFinite { what: "reward", index: 3 }.to_string(), "reward is not finite at 3");
    }

    /// A terminal array of the wrong length is refused in both directions, and an infinite reward
    /// is refused like a `NaN`. The suite's only length refusal was a short `next`, and its only
    /// non-finite reward was a `NaN` — which `is_nan()` alone already rejects, so the screen could
    /// lose its infinities and every fixture would still pass.
    #[test]
    fn a_task_refuses_a_mismatched_terminal_array_and_a_non_finite_reward_of_either_kind() {
        assert_eq!(
            Task::new(3, 1, vec![0, 1, 2], vec![0.0; 3], vec![false, false], 0.5).unwrap_err(),
            ReinforceError::Index { what: "terminal (length)", index: 2, count: 3 }
        );
        assert_eq!(
            Task::new(3, 1, vec![0, 1, 2], vec![0.0; 3], vec![false; 4], 0.5).unwrap_err(),
            ReinforceError::Index { what: "terminal (length)", index: 4, count: 3 }
        );
        assert_eq!(
            Task::new(2, 1, vec![0, 1], vec![0.0, f64::INFINITY], vec![false, true], 0.5).unwrap_err(),
            ReinforceError::NonFinite { what: "reward", index: 1 }
        );
        assert_eq!(
            Task::new(2, 1, vec![0, 1], vec![f64::NEG_INFINITY, 0.0], vec![false, true], 0.5).unwrap_err(),
            ReinforceError::NonFinite { what: "reward", index: 0 }
        );
    }

    /// The chain's two tables, literally. The suite read the chain only through the always-right
    /// policy and through episodes that stop on entering the terminal state, so the left column and
    /// the terminal row — two of the four things `chain` writes — were never read by any assertion.
    #[test]
    fn the_chain_writes_this_exact_transition_and_reward_table() {
        // s: left, right. State 3 is terminal and maps to itself both ways; the unit reward is on
        // the step INTO it, from state 2.
        let four = Task::chain(4, 0.9).unwrap();
        assert_eq!(four.next, vec![0, 1, 0, 2, 1, 3, 3, 3]);
        assert_eq!(four.reward, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(four.terminal, vec![false, false, false, true]);
        let two = Task::chain(2, 0.5).unwrap();
        assert_eq!(two.next, vec![0, 1, 1, 1]);
        assert_eq!(two.reward, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(two.terminal, vec![false, true]);
    }

    /// Value iteration pins a terminal state at zero however well its own row pays. On the chain
    /// the terminal row pays nothing into itself, so a sweep that backs terminal states up lands on
    /// zero there anyway and every fixture in the suite agrees with it.
    #[test]
    fn value_iteration_pins_a_terminal_state_at_zero_however_well_its_own_row_pays() {
        // One action. State 1 is terminal and its row pays 1.0 into itself, which a sweep that
        // backed it up would value at 1/(1 − γ) = 2 and pass back to state 0 as 0.5 + 0.5·2 = 1.5.
        let task = Task::new(2, 1, vec![1, 1], vec![0.5, 1.0], vec![false, true], 0.5).unwrap();
        assert_eq!(task.values_of(&[0, 0], 1e-14).unwrap(), vec![0.5, 0.0]);
    }

    /// `active` counts the traces above the floor it was handed, not the ones that are merely
    /// non-zero, and `reset` reaches synapse 0 as well as the rest. The suite called `active` with
    /// a trace of 0.19 against a floor of 1e-6 and with an all-zero vector — never with a trace
    /// between the two — and it spiked synapse 1 rather than synapse 0 before resetting.
    #[test]
    fn the_active_count_respects_its_floor_and_a_reset_reaches_the_first_synapse() {
        let mut e = Eligibility::new(3, 0.5).unwrap();
        e.spike(0).unwrap();
        e.spike(2).unwrap();
        for _ in 0..30 {
            e.step();
        }
        // Halving is exact in binary, so this is 2^-30 = 9.3e-10 to the last bit: non-zero, and
        // three orders of magnitude below the floor.
        assert_eq!(e.e, vec![0.5f64.powi(30), 0.0, 0.5f64.powi(30)]);
        assert_eq!(e.active(1e-6), 0);
        assert_eq!(e.active(0.0), 2);
        e.reset();
        assert_eq!(e.e, vec![0.0, 0.0, 0.0]);
    }

    /// `λ = 1` is inside the closed interval the critic documents, and it leaves the trace decaying
    /// by the discount alone. Every fixture used λ ∈ {0, 0.5, 0.8} and the only λ the suite saw
    /// refused was 1.5, which a half-open guard refuses too.
    #[test]
    fn a_lambda_of_exactly_one_is_accepted_and_leaves_the_trace_decaying_by_the_discount() {
        let c = Critic::new(4, 0.1, 0.9, 1.0).unwrap();
        assert_eq!(c.trace.decay, 0.9, "γλ at λ = 1 is γ");
        assert!(matches!(
            Critic::new(4, 0.1, 0.9, 1.0 + f64::EPSILON),
            Err(ReinforceError::OutOfRange { what: "lambda", .. })
        ));
        // γ stays half-open: at γ = λ = 1 the trace would never fade at all.
        assert!(matches!(Critic::new(4, 0.1, 1.0, 1.0), Err(ReinforceError::OutOfRange { what: "gamma", .. })));
    }

    /// The critic refuses a successor one past its table whether or not that successor is flagged
    /// terminal. The suite's only index refusal was the state LEFT; the successors it passed were
    /// inside the table, and a terminal successor is never read, so an off-by-one there does not
    /// even panic — it returns an error-free update.
    #[test]
    fn the_critic_refuses_a_successor_one_past_its_table_even_when_it_is_terminal() {
        let mut c = Critic::new(3, 0.1, 0.9, 0.5).unwrap();
        assert_eq!(
            c.update(0, 0.0, 3, true).unwrap_err(),
            ReinforceError::Index { what: "next state", index: 3, count: 3 }
        );
        assert_eq!(
            c.update(0, 0.0, 3, false).unwrap_err(),
            ReinforceError::Index { what: "next state", index: 3, count: 3 }
        );
        assert_eq!(c.updates, 0, "a refused transition is not an update");
    }

    /// The weight change carries the SIZE of the trace: under one error, a synapse eligible at 0.5
    /// moves half as far as one eligible at 1. The suite read the trace only through runs that
    /// converge — `η δ` alone has the same fixed point as `η δ e` on the chain, since both stop
    /// when δ does — and through λ = 0, where the one eligible trace is exactly 1 and the factor
    /// cannot be seen.
    #[test]
    fn the_weight_change_is_the_error_times_the_trace_and_not_the_error_alone() {
        // η = 0.5, γ = 0.5, λ = 1 → the trace decays by 0.5 a step, so one step after its spike
        // state 0 is eligible at 0.5 while state 1 is eligible at 1.
        let mut c = Critic::new(3, 0.5, 0.5, 1.0).unwrap();
        assert_eq!(c.update(0, 0.0, 1, false).unwrap(), 0.0);
        assert_eq!(c.update(1, 1.0, 2, true).unwrap(), 1.0, "δ = 1 + γ·0 − V(1)");
        assert_eq!(c.v, vec![0.5 * 1.0 * 0.5, 0.5 * 1.0 * 1.0, 0.0], "η δ e per synapse");
        assert_eq!(c.touched, 3);
        assert_eq!(c.updates, 2);
        assert_eq!(c.touched_per_update(), Some(1.5));
    }

    /// `touched_per_update` is `None` only before the first update — an update that found nothing
    /// eligible is `Some(0.0)`, a different fact. No fixture raised the floor, and under the
    /// default 1e-6 the state just spiked is always eligible at 1 or more, so `touched == 0` and
    /// `updates == 0` were the same condition everywhere the suite looked.
    #[test]
    fn the_touched_mean_is_none_only_before_the_first_update_not_when_nothing_was_eligible() {
        let mut c = Critic::new(3, 0.5, 0.5, 0.0).unwrap();
        assert_eq!(c.touched_per_update(), None);
        c.floor = 2.0; // above the 1.0 a single spike deposits: nothing is eligible
        assert_eq!(c.update(0, 1.0, 1, true).unwrap(), 1.0);
        assert_eq!(c.touched, 0);
        assert_eq!(c.updates, 1);
        assert_eq!(c.touched_per_update(), Some(0.0));
        assert_eq!(c.v, vec![0.0, 0.0, 0.0], "an update that touches nothing moves nothing");
    }

    /// Ending an episode ZEROES both traces rather than decaying them one step: credit must not
    /// cross the seam between episodes. The suite ended episodes only inside runs whose verdict is
    /// a converged value table or a learned policy, and one decayed step of stale trace per episode
    /// is a perturbation those tolerances absorb.
    #[test]
    fn ending_an_episode_zeroes_the_critics_and_the_actors_traces() {
        let mut c = Critic::new(3, 0.5, 0.9, 0.8).unwrap();
        c.update(0, 1.0, 1, false).unwrap();
        assert_eq!(c.trace.e, vec![1.0, 0.0, 0.0]);
        c.end_episode();
        assert_eq!(c.trace.e, vec![0.0, 0.0, 0.0]);
        let mut a = Actor::new(2, 2, 0.5, 0.72).unwrap();
        let mut rng = Rng::new(11);
        a.act(0, &mut rng).unwrap();
        assert_eq!(a.trace.e[2..], [0.0, 0.0], "the act touched only state 0's row");
        assert!(a.trace.e[0] != 0.0 && a.trace.e[1] != 0.0, "and left something there to clear");
        a.end_episode();
        assert_eq!(a.trace.e, vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// The softmax subtracts the LARGEST preference before exponentiating, so a preference no `exp`
    /// can represent still gives a policy. The suite's actors started at θ = 0 and learned
    /// preferences of a few units, where subtracting the smallest is as good as subtracting the
    /// largest — the shift is an overflow guard, and only an overflow reads it.
    #[test]
    fn the_softmax_subtracts_the_largest_preference_so_a_huge_one_does_not_overflow() {
        let mut a = Actor::new(1, 3, 0.1, 0.5).unwrap();
        // exp(800) is +∞ in f64 — the ceiling is exp(709.78) — and exp(−800) underflows to 0, so
        // shifting by the smallest preference gives ∞/∞ = NaN where shifting by the largest gives
        // the policy of the differences.
        a.theta = vec![800.0, 799.0, 0.0];
        let p = a.policy(0).unwrap();
        assert!(p.iter().all(|x| x.is_finite()), "policy {p:?}");
        let z = 1.0 + (-1.0f64).exp();
        assert_eq!(p, vec![1.0 / z, (-1.0f64).exp() / z, 0.0]);
    }

    /// The actor's eligibility ACCUMULATES over a revisit inside one episode: the second visit's
    /// gradient is added to the decayed first, not written over it. Every fixture walked the chain
    /// left to right, visiting each state once per episode, and a single visit is the one case in
    /// which `=` and `+=` agree.
    #[test]
    fn the_actors_eligibility_accumulates_when_a_state_is_visited_twice() {
        let mut a = Actor::new(1, 2, 0.1, 0.5).unwrap();
        a.theta = vec![4.0f64.ln(), 0.0]; // π = (0.8, 0.2), so the two gradients differ
        let mut rng = Rng::new(3);
        a.act(0, &mut rng).unwrap();
        let first = a.trace.e.clone();
        assert!(first[0] != 0.0 && first[1] != 0.0, "both synapses carry a gradient: {first:?}");
        let again = a.act(0, &mut rng).unwrap();
        let p = a.policy(0).unwrap(); // θ is untouched: `act` deposits, it does not learn
        for k in 0..2 {
            let g = if k == again { 1.0 - p[k] } else { -p[k] };
            assert_eq!(a.trace.e[k], first[k] * 0.5 + g, "synapse {k}");
        }
    }

    /// The actor's learning rate scales its preference change. The suite read the actor only
    /// through a policy that had converged past P(right) > 0.9 and a late average return, and a
    /// policy gradient taken at a rate of 1 rather than 0.5 arrives there too — sooner, which no
    /// assertion measured.
    #[test]
    fn the_actors_learning_rate_scales_the_preference_change() {
        let mut a = Actor::new(1, 2, 0.5, 0.5).unwrap();
        a.trace.e = vec![1.0, -1.0];
        a.reinforce(2.0);
        assert_eq!(a.theta, vec![0.5 * 2.0, -(0.5 * 2.0)], "η δ e per synapse, with e = ±1");
    }

    /// The first reward of an episode is UNDISCOUNTED: the return is `Σ γ^t r_t` with `t` starting
    /// at zero. The suite compared a late average return against `V(0) = 0.656` at a tolerance of
    /// 0.1, and one spurious factor of γ = 0.9 moves it by 0.066 — inside that tolerance.
    #[test]
    fn the_first_reward_of_an_episode_is_undiscounted() {
        // One action per state, so the actor's sampling cannot vary the trajectory: 0 → 1 → 2 with
        // a unit reward on each of the two steps and γ = 0.5.
        let task = Task::new(3, 1, vec![1, 2, 2], vec![1.0, 1.0, 0.0], vec![false, false, true], 0.5).unwrap();
        let mut c = Critic::new(3, 0.1, 0.5, 0.0).unwrap();
        let mut a = Actor::new(3, 1, 0.1, 0.0).unwrap();
        let mut rng = Rng::new(5);
        let (ret, steps) = episode(&task, &mut c, &mut a, 0, 10, &mut rng).unwrap();
        assert_eq!(steps, 2);
        assert_eq!(ret, 1.0 + 0.5 * 1.0);
    }

    /// The terminal flag an episode hands the critic is the state ENTERED, not the state left. The
    /// loop runs only while the state left is non-terminal, so `terminal[s]` is false at every
    /// call — and on the chain the critic's cell for the terminal state is never written, so it
    /// holds the 0 the flag would have supplied and the substitution is invisible.
    #[test]
    fn the_terminal_flag_the_episode_passes_is_the_state_entered() {
        let task = Task::new(2, 1, vec![1, 1], vec![1.0, 0.0], vec![false, true], 0.5).unwrap();
        let mut c = Critic::new(2, 0.5, 0.5, 0.0).unwrap();
        c.v[1] = 4.0; // a terminal state whose stored value is not zero
        let mut a = Actor::new(2, 1, 0.1, 0.0).unwrap();
        let mut rng = Rng::new(5);
        let (ret, steps) = episode(&task, &mut c, &mut a, 0, 10, &mut rng).unwrap();
        assert_eq!((ret, steps), (1.0, 1));
        // δ = 1 + γ·0 − V(0) = 1, so V(0) moves by η δ e = 0.5·1·1. Bootstrapping off V(1) = 4
        // instead would make δ = 1 + 0.5·4 = 3 and V(0) = 1.5.
        assert_eq!(c.v, vec![0.5, 4.0]);
    }

    /// An episode refuses a start exactly one past the last state. The suite's only start refusal
    /// was 9 against 3 states, which an off-by-one in that guard still catches.
    #[test]
    fn an_episode_refuses_a_start_exactly_one_past_the_last_state() {
        let task = Task::chain(3, 0.9).unwrap();
        let mut c = Critic::new(3, 0.1, 0.9, 0.5).unwrap();
        let mut a = Actor::new(3, 2, 0.1, 0.5).unwrap();
        let mut rng = Rng::new(1);
        assert_eq!(
            episode(&task, &mut c, &mut a, 3, 10, &mut rng).unwrap_err(),
            ReinforceError::Index { what: "start", index: 3, count: 3 }
        );
    }
}
