//! Learning without weight transport: feedback alignment and direct feedback alignment, beside
//! the backpropagation they replace.
//!
//! # What the mechanism is
//!
//! Backpropagation sends the output error back through the TRANSPOSE of each forward weight
//! matrix: `δ_l = (W_{l+1}ᵀ δ_{l+1}) ⊙ f′(a_l)`. A physical network has no way to do that — the
//! backward pathway would need, at every synapse, a copy of a weight that lives at a different
//! synapse (the *weight transport* problem). Lillicrap, Cownden, Tweed and Akerman (*Random
//! synaptic feedback weights support error backpropagation for deep learning*, Nature
//! Communications 7:13276, 2016) found the transpose is not needed: replace `W_{l+1}ᵀ` with a
//! FIXED RANDOM matrix `B_{l+1}` and the network still learns, because the forward weights come
//! to align with the feedback they are given — **feedback alignment**. Nøkland (*Direct feedback
//! alignment provides learning in deep neural networks*, `NeurIPS` 2016) removed the backward chain
//! altogether: every hidden layer receives the OUTPUT error through its own fixed random matrix,
//! `δ_l = (D_l e) ⊙ f′(a_l)` — **direct feedback alignment** — so no layer waits for the one
//! above it.
//!
//! # Why it is in a neuromorphic crate
//!
//! On-chip learning is where weight transport is not a modelling objection but a wiring cost: a
//! symmetric backward pathway doubles the synaptic memory traffic, and a random one is a fixed
//! matrix that never has to be written. Direct feedback alignment is also the structure of
//! e-prop's broadcast learning signal ([`crate::eprop`]) — a fixed random projection of the
//! output error delivered to every neuron — so this module is that rule's feed-forward limit.
//!
//! # The closed forms this module is checked against
//!
//! - **Backpropagation is the gradient**: every weight and bias against central finite
//!   differences of the loss.
//! - **Feedback alignment with `B = Wᵀ` IS backpropagation**, entry for entry; with any `B` the
//!   OUTPUT layer's update is still the true gradient, because no feedback matrix enters it.
//! - **With one hidden layer, direct and sequential feedback are the same rule**: given the same
//!   matrix, [`Rule::Direct`] and [`Rule::Alignment`] return identical updates.
//! - **In a linear network, direct feedback through the product of the downstream transposes IS
//!   backpropagation**: `D_l = W_{l+1}ᵀ W_{l+2}ᵀ ⋯ W_Lᵀ` — the chain collapsed into one matrix,
//!   which is what "direct" means. (With a nonlinearity between, the collapsed chain would need
//!   the intermediate `f′`, and no fixed matrix can supply it.)
//! - **Alignment.** At a random start the hidden update is unrelated to the gradient; after
//!   training on a teacher network the layer nearest the output points about 60° from it (cosine
//!   near 0.5) under either rule, and the loss has fallen seventy-fold. Alignment arrives from
//!   the output downward: the layer below is, after the same training, still unaligned under
//!   sequential feedback (its cosine passed through −0.1 on the way) and at 0.37 under direct.
//!   All of this is measured at this module's seeds, and labelled so in the test.
//!
//! # What this module has NOT reproduced
//!
//! - The papers' benchmarks (MNIST, CIFAR), convolutional layers — where feedback alignment is
//!   known to struggle — and sign-symmetric or learned-feedback variants (Kolen–Pollack, weight
//!   mirrors).
//! - A proof of alignment. Lillicrap and colleagues prove it for a linear network with one hidden
//!   layer under conditions this module does not check; here it is measured.

use crate::rng::Rng;
use core::fmt;

/// The most weights one layer may hold.
pub const MAX_WEIGHTS: usize = 1 << 24;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentError {
    /// Fewer than two layer sizes, a layer of zero units, or a layer past [`MAX_WEIGHTS`].
    BadShape,
    /// A vector of the wrong length.
    Shape {
        /// Which vector.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A `NaN` or infinity.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
    /// [`Rule::Direct`] was asked of a network whose direct matrices do not reach from the output.
    NoDirectFeedback,
}

impl fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("a network needs at least two layers, each of at least one unit"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
            Self::NonFinite { what } => write!(f, "{what} is not finite"),
            Self::NoDirectFeedback => f.write_str("the direct feedback matrices do not match the network's shape"),
        }
    }
}

impl std::error::Error for AlignmentError {}

/// The hidden nonlinearity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `tanh`.
    Tanh,
    /// The identity: a linear network, for the closed forms that need one.
    Linear,
}

impl Activation {
    fn f(self, a: f64) -> f64 {
        match self {
            Self::Tanh => a.tanh(),
            Self::Linear => a,
        }
    }

    /// The derivative, from the OUTPUT `h = f(a)`.
    fn slope(self, h: f64) -> f64 {
        match self {
            Self::Tanh => 1.0 - h * h,
            Self::Linear => 1.0,
        }
    }
}

/// How the output error reaches a hidden layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Through the transposes of the forward weights: the true gradient.
    Backprop,
    /// Through the fixed matrices [`Mlp::feedback`], layer by layer.
    Alignment,
    /// Straight from the output through the fixed matrices [`Mlp::direct`].
    Direct,
}

/// One layer's update: the estimate of `∂L/∂W` and `∂L/∂b` the rule produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// `δ hᵀ`, row-major, shaped like the layer's weights.
    pub w: Vec<f64>,
    /// `δ`.
    pub b: Vec<f64>,
}

/// A multilayer perceptron with a linear output and loss `½ |y − t|²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    /// Units per layer, input first.
    pub sizes: Vec<usize>,
    /// The hidden nonlinearity.
    pub activation: Activation,
    /// `w[l]`, row-major `sizes[l+1] × sizes[l]`.
    pub w: Vec<Vec<f64>>,
    /// `b[l]`, `sizes[l+1]` long.
    pub b: Vec<Vec<f64>>,
    /// `feedback[l]`, row-major `sizes[l+1] × sizes[l+2]`: what [`Rule::Alignment`] uses in place
    /// of `w[l+1]ᵀ`. One per hidden layer.
    pub feedback: Vec<Vec<f64>>,
    /// `direct[l]`, row-major `sizes[l+1] × n_out`: what [`Rule::Direct`] multiplies the output
    /// error by. One per hidden layer.
    pub direct: Vec<Vec<f64>>,
}

fn uniform(rng: &mut Rng, rows: usize, cols: usize, scale: f64) -> Vec<f64> {
    (0..rows * cols).map(|_| scale * (2.0 * rng.next_f64() - 1.0)).collect()
}

impl Mlp {
    /// A random network. Forward weights are uniform in `±1/√fan_in`; both kinds of feedback
    /// matrix are uniform in `±1/√(their input width)`, and are never changed by learning.
    ///
    /// # Errors
    ///
    /// [`AlignmentError::BadShape`].
    pub fn random(sizes: &[usize], activation: Activation, seed: u64) -> Result<Self, AlignmentError> {
        if sizes.len() < 2 || sizes.contains(&0) || sizes.windows(2).any(|p| p[0].saturating_mul(p[1]) > MAX_WEIGHTS) {
            return Err(AlignmentError::BadShape);
        }
        let mut rng = Rng::new(seed);
        let layers = sizes.len() - 1;
        let n_out = sizes[layers];
        let w = (0..layers).map(|l| uniform(&mut rng, sizes[l + 1], sizes[l], 1.0 / (sizes[l] as f64).sqrt())).collect();
        let b = (0..layers).map(|l| vec![0.0; sizes[l + 1]]).collect();
        let feedback = (0..layers - 1).map(|l| uniform(&mut rng, sizes[l + 1], sizes[l + 2], 1.0 / (sizes[l + 2] as f64).sqrt())).collect();
        let direct = (0..layers - 1).map(|l| uniform(&mut rng, sizes[l + 1], n_out, 1.0 / (n_out as f64).sqrt())).collect();
        Ok(Self { sizes: sizes.to_vec(), activation, w, b, feedback, direct })
    }

    fn layers(&self) -> usize {
        self.sizes.len() - 1
    }

    /// The activity of every layer, input first; the last entry is the output.
    ///
    /// # Errors
    ///
    /// [`AlignmentError::Shape`] or [`AlignmentError::NonFinite`] for a bad input.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, AlignmentError> {
        if x.len() != self.sizes[0] {
            return Err(AlignmentError::Shape { what: "x", got: x.len(), want: self.sizes[0] });
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(AlignmentError::NonFinite { what: "x" });
        }
        let mut h = vec![x.to_vec()];
        for l in 0..self.layers() {
            let (rows, cols) = (self.sizes[l + 1], self.sizes[l]);
            let last = l + 1 == self.layers();
            let below = &h[l];
            let next = (0..rows)
                .map(|i| {
                    let a = self.b[l][i] + self.w[l][i * cols..(i + 1) * cols].iter().zip(below).map(|(w, h)| w * h).sum::<f64>();
                    if last { a } else { self.activation.f(a) }
                })
                .collect();
            h.push(next);
        }
        Ok(h)
    }

    /// `½ |y − t|²`.
    ///
    /// # Errors
    ///
    /// As [`Mlp::forward`], and for a target of the wrong length.
    pub fn loss(&self, x: &[f64], target: &[f64]) -> Result<f64, AlignmentError> {
        let h = self.forward(x)?;
        let y = &h[self.layers()];
        if target.len() != y.len() {
            return Err(AlignmentError::Shape { what: "target", got: target.len(), want: y.len() });
        }
        Ok(0.5 * y.iter().zip(target).map(|(y, t)| (y - t) * (y - t)).sum::<f64>())
    }

    /// Every layer's update under `rule`, input-side layer first. Under [`Rule::Backprop`] these
    /// are the gradient of [`Mlp::loss`]; under the other two only the last layer's is.
    ///
    /// # Errors
    ///
    /// As [`Mlp::loss`]; [`AlignmentError::NoDirectFeedback`] or [`AlignmentError::Shape`] if a
    /// feedback matrix the rule needs has been given the wrong size.
    pub fn updates(&self, x: &[f64], target: &[f64], rule: Rule) -> Result<Vec<Update>, AlignmentError> {
        let layers = self.layers();
        let h = self.forward(x)?;
        let n_out = self.sizes[layers];
        if target.len() != n_out {
            return Err(AlignmentError::Shape { what: "target", got: target.len(), want: n_out });
        }
        if !target.iter().all(|v| v.is_finite()) {
            return Err(AlignmentError::NonFinite { what: "target" });
        }
        let error: Vec<f64> = h[layers].iter().zip(target).map(|(y, t)| y - t).collect();
        let mut delta = error.clone();
        let mut out = vec![Update { w: Vec::new(), b: Vec::new() }; layers];
        for l in (0..layers).rev() {
            let cols = self.sizes[l];
            out[l] = Update { w: delta.iter().flat_map(|d| h[l].iter().map(move |h| d * h)).collect(), b: delta.clone() };
            if l == 0 {
                break;
            }
            // What arrives at hidden layer l (activity h[l]), before its own slope.
            let arriving: Vec<f64> = match rule {
                Rule::Backprop => {
                    let rows = self.sizes[l + 1];
                    (0..cols).map(|j| (0..rows).map(|i| self.w[l][i * cols + j] * delta[i]).sum()).collect()
                }
                Rule::Alignment => {
                    let from = self.sizes[l + 1];
                    let m = &self.feedback[l - 1];
                    if m.len() != cols * from {
                        return Err(AlignmentError::Shape { what: "feedback", got: m.len(), want: cols * from });
                    }
                    (0..cols).map(|j| m[j * from..(j + 1) * from].iter().zip(&delta).map(|(m, d)| m * d).sum()).collect()
                }
                Rule::Direct => {
                    let m = self.direct.get(l - 1).ok_or(AlignmentError::NoDirectFeedback)?;
                    if m.len() != cols * n_out {
                        return Err(AlignmentError::NoDirectFeedback);
                    }
                    (0..cols).map(|j| m[j * n_out..(j + 1) * n_out].iter().zip(&error).map(|(m, e)| m * e).sum()).collect()
                }
            };
            delta = arriving.iter().zip(&h[l]).map(|(a, h)| a * self.activation.slope(*h)).collect();
        }
        Ok(out)
    }

    /// Move every weight and bias against its update: `w ← w − rate · update`.
    ///
    /// # Errors
    ///
    /// [`AlignmentError::Shape`] if the updates are not this network's, [`AlignmentError::NonFinite`]
    /// for a non-finite rate.
    pub fn apply(&mut self, updates: &[Update], rate: f64) -> Result<(), AlignmentError> {
        if !rate.is_finite() {
            return Err(AlignmentError::NonFinite { what: "rate" });
        }
        if updates.len() != self.layers() {
            return Err(AlignmentError::Shape { what: "updates", got: updates.len(), want: self.layers() });
        }
        for (l, u) in updates.iter().enumerate() {
            if u.w.len() != self.w[l].len() || u.b.len() != self.b[l].len() {
                return Err(AlignmentError::Shape { what: "update", got: u.w.len(), want: self.w[l].len() });
            }
        }
        for (l, u) in updates.iter().enumerate() {
            self.w[l].iter_mut().zip(&u.w).for_each(|(w, g)| *w -= rate * g);
            self.b[l].iter_mut().zip(&u.b).for_each(|(b, g)| *b -= rate * g);
        }
        Ok(())
    }
}

/// The cosine of the angle between two updates of the same shape; `None` if either is zero or
/// the lengths differ.
#[must_use]
pub fn cosine(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let dot: f64 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let (na, nb) = (a.iter().map(|a| a * a).sum::<f64>().sqrt(), b.iter().map(|b| b * b).sum::<f64>().sqrt());
    if na > 0.0 && nb > 0.0 { Some(dot / (na * nb)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(rng: &mut Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
    }

    #[test]
    fn backpropagation_is_the_gradient_of_the_loss() {
        let mut net = Mlp::random(&[4, 6, 5, 3], Activation::Tanh, 2).unwrap();
        let mut rng = Rng::new(9);
        for b in &mut net.b {
            *b = point(&mut rng, b.len());
        }
        let (x, t) = (point(&mut rng, 4), point(&mut rng, 3));
        let grads = net.updates(&x, &t, Rule::Backprop).unwrap();
        let h = 1e-6;
        let mut checked = 0;
        for l in 0..3 {
            for k in 0..net.w[l].len() {
                let mut probe = net.clone();
                probe.w[l][k] += h;
                let up = probe.loss(&x, &t).unwrap();
                probe.w[l][k] -= 2.0 * h;
                let fd = (up - probe.loss(&x, &t).unwrap()) / (2.0 * h);
                assert!((grads[l].w[k] - fd).abs() < 1e-8, "w[{l}][{k}]: {} against {fd}", grads[l].w[k]);
                checked += 1;
            }
            for k in 0..net.b[l].len() {
                let mut probe = net.clone();
                probe.b[l][k] += h;
                let up = probe.loss(&x, &t).unwrap();
                probe.b[l][k] -= 2.0 * h;
                let fd = (up - probe.loss(&x, &t).unwrap()) / (2.0 * h);
                assert!((grads[l].b[k] - fd).abs() < 1e-8, "b[{l}][{k}]: {} against {fd}", grads[l].b[k]);
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * 6 + 6 + 6 * 5 + 5 + 5 * 3 + 3);
        // And a step moves every parameter by exactly `-rate * gradient` — the weights AND the
        // biases, which a rule that quietly left the biases where they were would still pass a
        // gradient check.
        let before = net.clone();
        net.apply(&grads, 0.25).unwrap();
        let mut moved = 0;
        for l in 0..3 {
            for (k, g) in grads[l].w.iter().enumerate() {
                assert!((net.w[l][k] - (before.w[l][k] - 0.25 * g)).abs() < 1e-18);
                moved += usize::from(net.w[l][k] != before.w[l][k]);
            }
            for (k, g) in grads[l].b.iter().enumerate() {
                assert!((net.b[l][k] - (before.b[l][k] - 0.25 * g)).abs() < 1e-18);
                moved += usize::from(net.b[l][k] != before.b[l][k]);
            }
        }
        assert_eq!(moved, checked, "a step left a parameter where it was");
    }

    fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        (0..cols).flat_map(|j| (0..rows).map(move |i| m[i * cols + j])).collect()
    }

    #[test]
    fn feedback_through_the_transpose_is_backpropagation_and_the_output_layer_never_needs_it() {
        let mut net = Mlp::random(&[4, 6, 5, 3], Activation::Tanh, 3).unwrap();
        let mut rng = Rng::new(4);
        let (x, t) = (point(&mut rng, 4), point(&mut rng, 3));
        let bp = net.updates(&x, &t, Rule::Backprop).unwrap();
        let fa = net.updates(&x, &t, Rule::Alignment).unwrap();
        let dfa = net.updates(&x, &t, Rule::Direct).unwrap();
        // The last layer is the gradient under every rule …
        assert_eq!(fa[2], bp[2]);
        assert_eq!(dfa[2], bp[2]);
        // … and with random feedback the hidden ones are not.
        assert!(cosine(&fa[0].w, &bp[0].w).unwrap() < 0.9 && cosine(&dfa[0].w, &bp[0].w).unwrap() < 0.9);
        assert_ne!(fa[1], dfa[1]);
        // Give feedback alignment the transposes and it is backpropagation.
        net.feedback[0] = transpose(&net.w[1], 5, 6);
        net.feedback[1] = transpose(&net.w[2], 3, 5);
        let fa = net.updates(&x, &t, Rule::Alignment).unwrap();
        for l in 0..3 {
            for (a, b) in fa[l].w.iter().zip(&bp[l].w).chain(fa[l].b.iter().zip(&bp[l].b)) {
                assert!((a - b).abs() < 1e-15, "layer {l}: {a} against {b}");
            }
        }
    }

    #[test]
    fn with_one_hidden_layer_direct_and_sequential_feedback_are_one_rule() {
        let mut net = Mlp::random(&[5, 7, 2], Activation::Tanh, 6).unwrap();
        let mut rng = Rng::new(1);
        let (x, t) = (point(&mut rng, 5), point(&mut rng, 2));
        assert_ne!(net.updates(&x, &t, Rule::Direct).unwrap(), net.updates(&x, &t, Rule::Alignment).unwrap());
        net.direct[0] = net.feedback[0].clone();
        assert_eq!(net.updates(&x, &t, Rule::Direct).unwrap(), net.updates(&x, &t, Rule::Alignment).unwrap());
    }

    #[test]
    fn in_a_linear_network_direct_feedback_through_the_collapsed_chain_is_backpropagation() {
        let mut net = Mlp::random(&[4, 6, 5, 3], Activation::Linear, 8).unwrap();
        let mut rng = Rng::new(2);
        let (x, t) = (point(&mut rng, 4), point(&mut rng, 3));
        // D for the second hidden layer is W₃ᵀ (5 × 3); for the first it is W₂ᵀ W₃ᵀ (6 × 3).
        let w3t = transpose(&net.w[2], 3, 5);
        let w2t = transpose(&net.w[1], 5, 6);
        let chain: Vec<f64> = (0..6).flat_map(|i| (0..3).map(|j| (0..5).map(|k| w2t[i * 5 + k] * w3t[k * 3 + j]).sum::<f64>()).collect::<Vec<_>>()).collect();
        net.direct = vec![chain, w3t];
        let (bp, dfa) = (net.updates(&x, &t, Rule::Backprop).unwrap(), net.updates(&x, &t, Rule::Direct).unwrap());
        for l in 0..3 {
            for (a, b) in dfa[l].w.iter().zip(&bp[l].w) {
                assert!((a - b).abs() < 1e-14, "layer {l}: {a} against {b}");
            }
        }
        // The same matrices in a TANH network are not backpropagation: the collapsed chain has
        // lost the slopes of the layers it skipped.
        net.activation = Activation::Tanh;
        let (bp, dfa) = (net.updates(&x, &t, Rule::Backprop).unwrap(), net.updates(&x, &t, Rule::Direct).unwrap());
        let worst = dfa[0].w.iter().zip(&bp[0].w).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(worst > 1e-4, "with tanh between, the two agreed to {worst}");
    }

    /// Train on a fixed tanh teacher for 4,000 samples; returns the loss before and after, and
    /// for each hidden layer the mean per-sample cosine between the rule's update and the
    /// gradient's, before and after, all over a held-out batch.
    fn learn(rule: Rule) -> (f64, f64, [f64; 2], [f64; 2]) {
        let teacher = Mlp::random(&[8, 12, 4], Activation::Tanh, 100).unwrap();
        let mut net = Mlp::random(&[8, 16, 16, 4], Activation::Tanh, 200).unwrap();
        let mut rng = Rng::new(300);
        let batch: Vec<(Vec<f64>, Vec<f64>)> = (0..64)
            .map(|_| {
                let x = point(&mut rng, 8);
                let t = teacher.forward(&x).unwrap().pop().unwrap();
                (x, t)
            })
            .collect();
        let measure = |net: &Mlp| {
            let (mut loss, mut cos) = (0.0, [0.0f64; 2]);
            for (x, t) in &batch {
                loss += net.loss(x, t).unwrap() / batch.len() as f64;
                let (r, g) = (net.updates(x, t, rule).unwrap(), net.updates(x, t, Rule::Backprop).unwrap());
                for l in 0..2 {
                    cos[l] += cosine(&r[l].w, &g[l].w).unwrap() / batch.len() as f64;
                }
            }
            (loss, cos)
        };
        let (loss_before, cos_before) = measure(&net);
        for _ in 0..4_000 {
            let x = point(&mut rng, 8);
            let t = teacher.forward(&x).unwrap().pop().unwrap();
            let u = net.updates(&x, &t, rule).unwrap();
            net.apply(&u, 0.02).unwrap();
        }
        let (loss_after, cos_after) = measure(&net);
        (loss_before, loss_after, cos_before, cos_after)
    }

    #[test]
    fn random_feedback_learns_because_the_forward_weights_come_to_align_with_it() {
        let (before, after, _, cos) = learn(Rule::Backprop);
        assert!(after < 0.05 * before, "backprop: {before} → {after}");
        assert!((cos[0] - 1.0).abs() < 1e-12 && (cos[1] - 1.0).abs() < 1e-12);
        // MEASURED at these seeds. In the hidden layer nearest the output the update starts
        // unrelated to the gradient — a cosine inside the ±0.25 scatter of two random directions
        // in its error's 16 dimensions — and after training sits near 0.5, about 60° from it:
        // feedback alignment 0.22 → 0.53, direct 0.08 → 0.46. The loss falls seventy-fold.
        for rule in [Rule::Alignment, Rule::Direct] {
            let (before, after, cos_before, cos_after) = learn(rule);
            assert!(cos_before[1].abs() < 0.3, "{rule:?}: the starting cosine was already {}", cos_before[1]);
            assert!(cos_after[1] > 0.4, "{rule:?}: cosine {} → {}", cos_before[1], cos_after[1]);
            assert!(after < 0.05 * before, "{rule:?}: loss {before} → {after}");
            // The layer below aligns later and less: after the same training its cosine is still
            // inside the scatter under sequential feedback (0.06) and 0.37 under direct.
            assert!(cos_after[0] < cos_after[1], "{rule:?}: the deeper layer aligned more: {cos_after:?}");
        }
    }

    #[test]
    fn bad_shapes_and_values_are_refused() {
        assert_eq!(Mlp::random(&[4], Activation::Tanh, 1), Err(AlignmentError::BadShape));
        assert_eq!(Mlp::random(&[4, 0, 2], Activation::Tanh, 1), Err(AlignmentError::BadShape));
        assert_eq!(Mlp::random(&[1 << 13, 1 << 13], Activation::Tanh, 1), Err(AlignmentError::BadShape));
        let mut net = Mlp::random(&[3, 4, 2], Activation::Tanh, 1).unwrap();
        assert_eq!(net.forward(&[1.0, 2.0]), Err(AlignmentError::Shape { what: "x", got: 2, want: 3 }));
        assert_eq!(net.forward(&[1.0, f64::NAN, 0.0]), Err(AlignmentError::NonFinite { what: "x" }));
        assert_eq!(net.loss(&[1.0, 2.0, 3.0], &[0.0]), Err(AlignmentError::Shape { what: "target", got: 1, want: 2 }));
        assert_eq!(net.updates(&[1.0, 2.0, 3.0], &[0.0], Rule::Backprop), Err(AlignmentError::Shape { what: "target", got: 1, want: 2 }));
        assert_eq!(net.updates(&[1.0, 2.0, 3.0], &[0.0, f64::INFINITY], Rule::Backprop), Err(AlignmentError::NonFinite { what: "target" }));
        let good = net.updates(&[1.0, 2.0, 3.0], &[0.0, 1.0], Rule::Backprop).unwrap();
        let before = net.clone();
        assert_eq!(net.apply(&good, f64::NAN), Err(AlignmentError::NonFinite { what: "rate" }));
        assert_eq!(net.apply(&good[..1], 0.1), Err(AlignmentError::Shape { what: "updates", got: 1, want: 2 }));
        let mut short = good.clone();
        short[1].w.pop();
        assert!(net.apply(&short, 0.1).is_err());
        assert_eq!(net, before, "a refused update must not move the network");
        net.direct[0].pop();
        assert_eq!(net.updates(&[1.0, 2.0, 3.0], &[0.0, 1.0], Rule::Direct), Err(AlignmentError::NoDirectFeedback));
        net.direct.clear();
        assert_eq!(net.updates(&[1.0, 2.0, 3.0], &[0.0, 1.0], Rule::Direct), Err(AlignmentError::NoDirectFeedback));
        net.feedback[0].pop();
        assert!(matches!(net.updates(&[1.0, 2.0, 3.0], &[0.0, 1.0], Rule::Alignment), Err(AlignmentError::Shape { what: "feedback", .. })));
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 2.0]), Some(0.0));
        assert_eq!(cosine(&[3.0, 4.0], &[6.0, 8.0]), Some(1.0));
        assert!(cosine(&[0.0, 0.0], &[1.0, 1.0]).is_none() && cosine(&[1.0], &[1.0, 2.0]).is_none());
        assert!(AlignmentError::NoDirectFeedback.to_string().contains("direct"));
    }
}
