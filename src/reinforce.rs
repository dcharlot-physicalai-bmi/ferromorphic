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
}
