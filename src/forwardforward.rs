//! The forward-forward algorithm: two forward passes — one on real data, one on negative data —
//! and a layer-local objective, in place of a forward and a backward pass.
//!
//! # What the mechanism is
//!
//! Hinton (*The forward-forward algorithm: some preliminary investigations*, arXiv:2212.13345,
//! 2022) gives each layer its own objective: have high **goodness** — the sum of its squared
//! activities, `G = Σ h²` — on positive (real) data and low goodness on negative data. The
//! probability that an input is positive is `σ(G − θ)`, and the layer descends the logistic loss
//! of that, `softplus(θ − G)` on positives and `softplus(G − θ)` on negatives. No derivative
//! crosses a layer boundary.
//!
//! Two details make a stack of such layers work. Before a layer's activity is passed up it is
//! **length-normalised**, so the next layer sees only the DIRECTION of the activity vector: the
//! goodness the layer below achieved is exactly the information removed, which forces the next
//! layer to find its own evidence rather than read off its predecessor's. And for supervised
//! learning the label is written INTO the input — a one-hot code over the first few inputs — so
//! a positive example is an input with its right label and a negative one the same input with a
//! wrong label; to classify, each label is tried in turn and the one with the highest total
//! goodness wins.
//!
//! # Why it is in a neuromorphic crate
//!
//! Hinton's stated motive is hardware: a learning procedure that needs no backward pass needs
//! no stored activations, no symmetric weights and no exact knowledge of the forward
//! computation, so it can run on analogue and spiking substrates whose forward pass is not
//! differentiable or not even known. It belongs beside [`crate::decolle`] (a local loss through
//! a random readout), [`crate::alignment`] (a global error through random feedback) and
//! [`crate::equilibrium`] (two relaxations of one energy) as the fourth way this crate has of
//! learning without backpropagation.
//!
//! # The closed forms this module is checked against
//!
//! - **The update is the gradient of the layer's own loss**, weights and biases, against central
//!   finite differences, for positive and for negative data.
//! - **At the threshold the layer is undecided**: `G = θ` gives probability one half and a loss
//!   of `ln 2` under either label.
//! - **Normalisation removes the goodness.** Scaling a layer's weights and biases by any `c > 0`
//!   multiplies its goodness by `c²` and leaves what it passes up unchanged, because the
//!   rectifier is positively homogeneous — so nothing above it can tell.
//! - **Locality.** A layer's update is unchanged, bit for bit, by any change to the layers above.
//! - **It learns.** On a four-class task with the label written into the input, goodness-based
//!   classification of held-out data rises from chance; measured, and labelled.
//!
//! # What this module has NOT reproduced
//!
//! - The paper's MNIST and CIFAR results, its unsupervised negative data (hybrid images), its
//!   recurrent top-down variant, and its alternative goodness functions.
//! - Its classification rule's exclusion of the first layer's goodness; [`Network::classify`]
//!   sums every layer's.

use crate::rng::Rng;
use core::fmt;

/// The most weights one layer may hold.
pub const MAX_WEIGHTS: usize = 1 << 24;

/// What went wrong, named rather than guessed around.
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardError {
    /// A layer of zero width or past [`MAX_WEIGHTS`], fewer than two sizes, or more labels than
    /// inputs.
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
    /// A `NaN` or infinity, or a threshold or rate that is not a positive finite number.
    NonFinite {
        /// Which quantity.
        what: &'static str,
    },
    /// An input of zero length, which has no direction to normalise to.
    ZeroInput,
    /// A label past the number of classes.
    BadLabel {
        /// The label supplied.
        label: usize,
        /// How many classes there are.
        classes: usize,
    },
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("a network needs an input and at least one layer, each of at least one unit, and no more labels than inputs"),
            Self::Shape { what, got, want } => write!(f, "{what} has {got} entries, not {want}"),
            Self::NonFinite { what } => write!(f, "{what} is not a finite number in range"),
            Self::ZeroInput => f.write_str("an all-zero input has no direction to normalise to"),
            Self::BadLabel { label, classes } => write!(f, "label {label} of {classes} classes"),
        }
    }
}

impl std::error::Error for ForwardError {}

/// `ln(1 + eˣ)` without overflow.
#[must_use]
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
}

/// A vector scaled to unit length.
///
/// # Errors
///
/// [`ForwardError::NonFinite`] or [`ForwardError::ZeroInput`].
pub fn normalised(x: &[f64]) -> Result<Vec<f64>, ForwardError> {
    if !x.iter().all(|v| v.is_finite()) {
        return Err(ForwardError::NonFinite { what: "x" });
    }
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm > 0.0) {
        return Err(ForwardError::ZeroInput);
    }
    Ok(x.iter().map(|v| v / norm).collect())
}

/// One layer's update: `∂L/∂W` and `∂L/∂b` of its own loss.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Row-major, `n × n_in`.
    pub w: Vec<f64>,
    /// `n` long.
    pub b: Vec<f64>,
}

/// One forward-forward layer: rectified linear units on a length-normalised input.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Inputs.
    pub n_in: usize,
    /// Units.
    pub n: usize,
    /// Weights, row-major `n × n_in`.
    pub w: Vec<f64>,
    /// Biases.
    pub b: Vec<f64>,
    /// The goodness threshold `θ`.
    pub theta: f64,
}

impl Layer {
    /// A random layer: weights uniform in `±1/√n_in`, zero biases.
    ///
    /// # Errors
    ///
    /// [`ForwardError::BadShape`]; [`ForwardError::NonFinite`] for a threshold that is not
    /// positive and finite.
    pub fn random(n_in: usize, n: usize, theta: f64, rng: &mut Rng) -> Result<Self, ForwardError> {
        if n_in == 0 || n == 0 || n_in.saturating_mul(n) > MAX_WEIGHTS {
            return Err(ForwardError::BadShape);
        }
        if !(theta > 0.0) || !theta.is_finite() {
            return Err(ForwardError::NonFinite { what: "theta" });
        }
        let scale = 1.0 / (n_in as f64).sqrt();
        let w = (0..n * n_in).map(|_| scale * (2.0 * rng.next_f64() - 1.0)).collect();
        Ok(Self { n_in, n, w, b: vec![0.0; n], theta })
    }

    /// The layer's activity `h = relu(W x̂ + b)` on the length-normalised input.
    ///
    /// # Errors
    ///
    /// [`ForwardError::Shape`], and as [`normalised`].
    pub fn activity(&self, x: &[f64]) -> Result<Vec<f64>, ForwardError> {
        if x.len() != self.n_in {
            return Err(ForwardError::Shape { what: "x", got: x.len(), want: self.n_in });
        }
        let unit = normalised(x)?;
        Ok((0..self.n).map(|i| (self.b[i] + self.w[i * self.n_in..(i + 1) * self.n_in].iter().zip(&unit).map(|(w, x)| w * x).sum::<f64>()).max(0.0)).collect())
    }

    /// The goodness `Σ h²` of an input.
    ///
    /// # Errors
    ///
    /// As [`Layer::activity`].
    pub fn goodness(&self, x: &[f64]) -> Result<f64, ForwardError> {
        Ok(self.activity(x)?.iter().map(|h| h * h).sum())
    }

    /// The probability the layer assigns to the input being positive, `σ(G − θ)`.
    ///
    /// # Errors
    ///
    /// As [`Layer::activity`].
    pub fn probability(&self, x: &[f64]) -> Result<f64, ForwardError> {
        Ok(sigmoid(self.goodness(x)? - self.theta))
    }

    /// The layer's loss on one input: `softplus(θ − G)` if it is positive data, `softplus(G − θ)`
    /// if negative.
    ///
    /// # Errors
    ///
    /// As [`Layer::activity`].
    pub fn loss(&self, x: &[f64], positive: bool) -> Result<f64, ForwardError> {
        let margin = self.goodness(x)? - self.theta;
        Ok(softplus(if positive { -margin } else { margin }))
    }

    /// The gradient of [`Layer::loss`]: `∂L/∂G · 2h ⊗ x̂`, with `∂L/∂G = −σ(θ − G)` on positive
    /// data and `σ(G − θ)` on negative.
    ///
    /// # Errors
    ///
    /// As [`Layer::activity`].
    pub fn update(&self, x: &[f64], positive: bool) -> Result<Update, ForwardError> {
        let h = self.activity(x)?;
        let unit = normalised(x)?;
        let margin = h.iter().map(|h| h * h).sum::<f64>() - self.theta;
        let d_goodness = if positive { -sigmoid(-margin) } else { sigmoid(margin) };
        // relu′ is carried by h itself: an inactive unit has h = 0 and contributes nothing.
        let b: Vec<f64> = h.iter().map(|h| d_goodness * 2.0 * h).collect();
        let w = b.iter().flat_map(|d| unit.iter().map(move |x| d * x)).collect();
        Ok(Update { w, b })
    }

    /// Move the weights against an update.
    ///
    /// # Errors
    ///
    /// [`ForwardError::Shape`] if the update is not this layer's; [`ForwardError::NonFinite`] for a
    /// non-finite rate.
    pub fn apply(&mut self, update: &Update, rate: f64) -> Result<(), ForwardError> {
        if !rate.is_finite() {
            return Err(ForwardError::NonFinite { what: "rate" });
        }
        if update.w.len() != self.w.len() || update.b.len() != self.b.len() {
            return Err(ForwardError::Shape { what: "update", got: update.w.len(), want: self.w.len() });
        }
        self.w.iter_mut().zip(&update.w).for_each(|(w, g)| *w -= rate * g);
        self.b.iter_mut().zip(&update.b).for_each(|(b, g)| *b -= rate * g);
        Ok(())
    }
}

/// A stack of forward-forward layers with the label written into the first `classes` inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    /// The layers, input side first.
    pub layers: Vec<Layer>,
    /// How many of the leading inputs hold the one-hot label.
    pub classes: usize,
}

impl Network {
    /// A random network. `sizes` is the input width (label slots included) and then each layer's
    /// width.
    ///
    /// # Errors
    ///
    /// [`ForwardError::BadShape`]; as [`Layer::random`].
    pub fn random(sizes: &[usize], classes: usize, theta: f64, seed: u64) -> Result<Self, ForwardError> {
        if sizes.len() < 2 || classes == 0 || classes >= sizes[0] {
            return Err(ForwardError::BadShape);
        }
        let mut rng = Rng::new(seed);
        let layers = sizes.windows(2).map(|p| Layer::random(p[0], p[1], theta, &mut rng)).collect::<Result<_, _>>()?;
        Ok(Self { layers, classes })
    }

    /// `features` with `label` written in front of them as a one-hot code.
    ///
    /// # Errors
    ///
    /// [`ForwardError::BadLabel`] or [`ForwardError::Shape`].
    pub fn labelled(&self, features: &[f64], label: usize) -> Result<Vec<f64>, ForwardError> {
        if label >= self.classes {
            return Err(ForwardError::BadLabel { label, classes: self.classes });
        }
        let want = self.layers[0].n_in - self.classes;
        if features.len() != want {
            return Err(ForwardError::Shape { what: "features", got: features.len(), want });
        }
        let mut x = vec![0.0; self.classes];
        x[label] = 1.0;
        x.extend_from_slice(features);
        Ok(x)
    }

    /// What each layer is given for input `x`: `x` itself, then each layer's activity. (Each
    /// layer normalises what it is given, which is where the goodness is removed.)
    ///
    /// # Errors
    ///
    /// As [`Layer::activity`]; [`ForwardError::ZeroInput`] if a layer falls entirely silent, since
    /// the one above it then has nothing to read.
    pub fn inputs(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, ForwardError> {
        let mut inputs = vec![x.to_vec()];
        for layer in &self.layers[..self.layers.len() - 1] {
            let h = layer.activity(&inputs[inputs.len() - 1])?;
            inputs.push(h);
        }
        Ok(inputs)
    }

    /// The goodness of every layer on `x`.
    ///
    /// # Errors
    ///
    /// As [`Network::inputs`].
    pub fn goodness(&self, x: &[f64]) -> Result<Vec<f64>, ForwardError> {
        let inputs = self.inputs(x)?;
        self.layers.iter().zip(&inputs).map(|(layer, input)| layer.goodness(input)).collect()
    }

    /// Every layer's own update on `x`.
    ///
    /// # Errors
    ///
    /// As [`Network::inputs`].
    pub fn updates(&self, x: &[f64], positive: bool) -> Result<Vec<Update>, ForwardError> {
        let inputs = self.inputs(x)?;
        self.layers.iter().zip(&inputs).map(|(layer, input)| layer.update(input, positive)).collect()
    }

    /// One positive pass with the right label and one negative pass with `wrong`, each layer
    /// descending its own loss.
    ///
    /// # Errors
    ///
    /// As [`Network::updates`] and [`Layer::apply`]; [`ForwardError::BadLabel`] if `wrong` is the
    /// right label.
    pub fn train(&mut self, features: &[f64], label: usize, wrong: usize, rate: f64) -> Result<(), ForwardError> {
        if wrong == label {
            return Err(ForwardError::BadLabel { label: wrong, classes: self.classes });
        }
        for (x, positive) in [(self.labelled(features, label)?, true), (self.labelled(features, wrong)?, false)] {
            let updates = self.updates(&x, positive)?;
            for (layer, update) in self.layers.iter_mut().zip(&updates) {
                layer.apply(update, rate)?;
            }
        }
        Ok(())
    }

    /// The label under which the network's total goodness is highest.
    ///
    /// # Errors
    ///
    /// As [`Network::goodness`].
    pub fn classify(&self, features: &[f64]) -> Result<usize, ForwardError> {
        let mut best = (0usize, f64::NEG_INFINITY);
        for label in 0..self.classes {
            let total: f64 = self.goodness(&self.labelled(features, label)?)?.iter().sum();
            if total > best.1 {
                best = (label, total);
            }
        }
        Ok(best.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(rng: &mut Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
    }

    #[test]
    fn the_update_is_the_gradient_of_the_layers_own_loss() {
        let mut rng = Rng::new(3);
        let mut layer = Layer::random(7, 9, 0.4, &mut rng).unwrap();
        layer.b = point(&mut rng, 9).iter().map(|b| 0.3 * b).collect();
        let x = point(&mut rng, 7);
        let h = layer.activity(&x).unwrap();
        let active = h.iter().filter(|&&h| h > 0.0).count();
        assert!((1..=8).contains(&active), "{active} of 9 units active: both branches of the rectifier are needed");
        assert!(h.iter().all(|&h| h == 0.0 || h > 1e-3), "a unit sits on the rectifier's kink");
        let step = 1e-6;
        for positive in [true, false] {
            let update = layer.update(&x, positive).unwrap();
            for k in 0..layer.w.len() {
                let mut probe = layer.clone();
                probe.w[k] += step;
                let up = probe.loss(&x, positive).unwrap();
                probe.w[k] -= 2.0 * step;
                let fd = (up - probe.loss(&x, positive).unwrap()) / (2.0 * step);
                assert!((update.w[k] - fd).abs() < 1e-9, "positive = {positive}, w[{k}]: {} against {fd}", update.w[k]);
            }
            for k in 0..layer.b.len() {
                let mut probe = layer.clone();
                probe.b[k] += step;
                let up = probe.loss(&x, positive).unwrap();
                probe.b[k] -= 2.0 * step;
                let fd = (up - probe.loss(&x, positive).unwrap()) / (2.0 * step);
                assert!((update.b[k] - fd).abs() < 1e-9, "positive = {positive}, b[{k}]: {} against {fd}", update.b[k]);
            }
            assert!(update.w.iter().any(|g| g.abs() > 1e-3));
        }
        // The two labels pull the goodness opposite ways.
        let (up, down) = (layer.update(&x, true).unwrap(), layer.update(&x, false).unwrap());
        assert!(up.b.iter().zip(&down.b).all(|(u, d)| u * d <= 0.0) && up.b.iter().any(|&u| u < 0.0));
        // A step moves every parameter by exactly `-rate * gradient`, biases included.
        let before = layer.clone();
        layer.apply(&up, 0.5).unwrap();
        let mut moved = 0;
        for (k, g) in up.w.iter().enumerate() {
            assert!((layer.w[k] - (before.w[k] - 0.5 * g)).abs() < 1e-18);
            moved += usize::from(layer.w[k] != before.w[k]);
        }
        for (k, g) in up.b.iter().enumerate() {
            assert!((layer.b[k] - (before.b[k] - 0.5 * g)).abs() < 1e-18);
            moved += usize::from(layer.b[k] != before.b[k]);
        }
        assert_eq!(moved, active * 7 + active, "only the active units move, and both their weights and their bias");
    }

    #[test]
    fn at_the_threshold_the_layer_is_undecided() {
        let mut rng = Rng::new(4);
        let mut layer = Layer::random(5, 6, 1.0, &mut rng).unwrap();
        let x = point(&mut rng, 5);
        layer.theta = layer.goodness(&x).unwrap();
        assert_eq!(layer.probability(&x).unwrap(), 0.5);
        assert!((layer.loss(&x, true).unwrap() - core::f64::consts::LN_2).abs() < 1e-15);
        assert!((layer.loss(&x, false).unwrap() - core::f64::consts::LN_2).abs() < 1e-15);
        // Away from it: by hand, G − θ = 0.5 gives σ = 0.6225 and losses ln(1 + e^{∓0.5}).
        layer.theta -= 0.5;
        assert!((layer.probability(&x).unwrap() - 1.0 / (1.0 + (-0.5f64).exp())).abs() < 1e-15);
        assert!((layer.loss(&x, true).unwrap() - (1.0 + (-0.5f64).exp()).ln()).abs() < 1e-15);
        assert!((layer.loss(&x, false).unwrap() - (1.0 + 0.5f64.exp()).ln()).abs() < 1e-15);
        assert!((softplus(800.0) - 800.0).abs() < 1e-12 && softplus(-800.0) == 0.0 && (softplus(0.0) - core::f64::consts::LN_2).abs() < 1e-16);
    }

    #[test]
    fn normalisation_hides_a_layers_goodness_from_the_layer_above() {
        let net = Network::random(&[10, 12, 8], 3, 1.0, 5).unwrap();
        let mut rng = Rng::new(6);
        let x = net.labelled(&point(&mut rng, 7), 1).unwrap();
        let mut louder = net.clone();
        louder.layers[0].w.iter_mut().for_each(|w| *w *= 3.0);
        louder.layers[0].b.iter_mut().for_each(|b| *b *= 3.0);
        let (quiet, loud) = (net.goodness(&x).unwrap(), louder.goodness(&x).unwrap());
        assert!(quiet[0] > 1e-3 && (loud[0] / quiet[0] - 9.0).abs() < 1e-12, "the first layer's goodness went {} → {}", quiet[0], loud[0]);
        assert!((loud[1] - quiet[1]).abs() < 1e-14 * quiet[1].max(1.0), "the second layer noticed: {} → {}", quiet[1], loud[1]);
        // The input's own length is invisible too.
        let scaled: Vec<f64> = x.iter().map(|v| 5.0 * v).collect();
        assert!((net.goodness(&scaled).unwrap()[0] - quiet[0]).abs() < 1e-14);
        let unit = normalised(&[3.0, -4.0]).unwrap();
        assert_eq!(unit, vec![0.6, -0.8]);
    }

    #[test]
    fn a_layers_update_does_not_depend_on_anything_above_it() {
        let net = Network::random(&[10, 12, 8, 6], 3, 1.0, 7).unwrap();
        let mut rng = Rng::new(8);
        let x = net.labelled(&point(&mut rng, 7), 2).unwrap();
        let whole = net.updates(&x, true).unwrap();
        assert!(whole.iter().all(|u| u.w.iter().any(|&g| g != 0.0)));
        let mut scrambled = net.clone();
        for layer in &mut scrambled.layers[1..] {
            layer.w.iter_mut().for_each(|w| *w = 0.5 - *w);
            layer.theta *= 2.0;
        }
        let other = scrambled.updates(&x, true).unwrap();
        assert_eq!(other[0], whole[0]);
        assert_ne!(other[1], whole[1]);
        // And a layer's update DOES depend on the one below, through what it is given.
        let mut below = net.clone();
        below.layers[0].w.iter_mut().for_each(|w| *w = 0.5 - *w);
        assert_ne!(below.updates(&x, true).unwrap()[1], whole[1]);
        // What a layer is given IS the activity of the one below it, entry for entry — the only
        // thing the stack does between layers is hand it over (each layer normalises its own).
        let inputs = net.inputs(&x).unwrap();
        assert_eq!(inputs[0], x);
        for l in 0..net.layers.len() - 1 {
            assert_eq!(inputs[l + 1], net.layers[l].activity(&inputs[l]).unwrap(), "layer {l} handed on something else");
        }
        assert!(inputs[1].iter().enumerate().any(|(i, v)| *v != inputs[1][inputs[1].len() - 1 - i]), "the fixture is symmetric, so a reversal would not show");
    }

    #[test]
    fn with_the_label_in_the_input_goodness_learns_to_classify() {
        // Four classes, each a fixed ±1 prototype over 16 features, seen through unit noise.
        let mut rng = Rng::new(21);
        let prototypes: Vec<Vec<f64>> = (0..4).map(|_| (0..16).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect()).collect();
        let sample = |rng: &mut Rng| {
            let label = rng.below(4) as usize;
            let features: Vec<f64> = prototypes[label].iter().map(|p| p + 1.5 * (rng.next_f64() + rng.next_f64() - 1.0)).collect();
            (features, label)
        };
        let mut net = Network::random(&[20, 32, 32], 4, 1.0, 22).unwrap();
        let held_out: Vec<(Vec<f64>, usize)> = (0..200).map(|_| sample(&mut rng)).collect();
        let accuracy = |net: &Network| held_out.iter().filter(|(f, l)| net.classify(f).unwrap() == *l).count();
        let before = accuracy(&net);
        // Classification is the SUM over layers, and the sum is not the first layer's vote: on
        // this untrained network there is an input where the two disagree, and `classify` follows
        // the total.
        let disagreement = held_out.iter().find_map(|(f, _)| {
            let per_label: Vec<Vec<f64>> = (0..4).map(|l| net.goodness(&net.labelled(f, l).unwrap()).unwrap()).collect();
            let best = |score: &dyn Fn(&Vec<f64>) -> f64| (0..4).max_by(|&a, &b| score(&per_label[a]).total_cmp(&score(&per_label[b]))).unwrap();
            let (first, total) = (best(&|g: &Vec<f64>| g[0]), best(&|g: &Vec<f64>| g.iter().sum()));
            (first != total).then_some((f.clone(), total))
        });
        let (features, total) = disagreement.expect("no input where the first layer and the total disagree");
        assert_eq!(net.classify(&features).unwrap(), total);
        for _ in 0..3_000 {
            let (features, label) = sample(&mut rng);
            let wrong = (label + 1 + rng.below(3) as usize) % 4;
            net.train(&features, label, wrong, 0.03).unwrap();
        }
        let after = accuracy(&net);
        // MEASURED at these seeds; chance is 50 of 200.
        assert!(before < 100, "the untrained network already scored {before} of 200");
        assert!(after >= 180, "held-out accuracy {before} → {after} of 200");
    }

    #[test]
    fn bad_shapes_and_values_are_refused() {
        let mut rng = Rng::new(1);
        assert_eq!(Layer::random(0, 3, 1.0, &mut rng), Err(ForwardError::BadShape));
        assert_eq!(Layer::random(1 << 13, 1 << 13, 1.0, &mut rng), Err(ForwardError::BadShape));
        assert_eq!(Layer::random(3, 3, 0.0, &mut rng), Err(ForwardError::NonFinite { what: "theta" }));
        assert_eq!(Network::random(&[5], 2, 1.0, 1), Err(ForwardError::BadShape));
        assert_eq!(Network::random(&[5, 4], 5, 1.0, 1), Err(ForwardError::BadShape));
        assert_eq!(Network::random(&[5, 4], 0, 1.0, 1), Err(ForwardError::BadShape));
        let mut net = Network::random(&[5, 4, 3], 2, 1.0, 1).unwrap();
        assert_eq!(net.labelled(&[1.0, 2.0, 3.0], 2), Err(ForwardError::BadLabel { label: 2, classes: 2 }));
        assert_eq!(net.labelled(&[1.0, 2.0], 0), Err(ForwardError::Shape { what: "features", got: 2, want: 3 }));
        assert_eq!(net.labelled(&[1.0, 2.0, 3.0], 1).unwrap(), vec![0.0, 1.0, 1.0, 2.0, 3.0]);
        assert_eq!(net.goodness(&[0.0; 5]), Err(ForwardError::ZeroInput));
        assert_eq!(net.goodness(&[0.0; 4]), Err(ForwardError::Shape { what: "x", got: 4, want: 5 }));
        assert_eq!(net.goodness(&[1.0, f64::NAN, 0.0, 0.0, 0.0]), Err(ForwardError::NonFinite { what: "x" }));
        let before = net.clone();
        assert_eq!(net.train(&[1.0, 2.0, 3.0], 1, 1, 0.1), Err(ForwardError::BadLabel { label: 1, classes: 2 }));
        assert_eq!(net.train(&[1.0, 2.0, 3.0], 1, 0, f64::NAN), Err(ForwardError::NonFinite { what: "rate" }));
        assert_eq!(net, before, "a refused step must not move the network");
        let update = Update { w: vec![0.0; 3], b: vec![0.0; 4] };
        assert!(matches!(net.layers[0].apply(&update, 0.1), Err(ForwardError::Shape { .. })));
        // A silent layer leaves the one above with nothing to read, and that is said.
        net.layers[0].w.iter_mut().for_each(|w| *w = 0.0);
        net.layers[0].b.iter_mut().for_each(|b| *b = -1.0);
        assert_eq!(net.goodness(&[1.0, 0.0, 1.0, 2.0, 3.0]), Err(ForwardError::ZeroInput));
        assert!(ForwardError::ZeroInput.to_string().contains("direction"));
    }
}
