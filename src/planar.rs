//! Two-variable models read in the phase plane: `FitzHugh`'s Bonhoeffer–van der Pol neuron and the
//! Wilson–Cowan excitatory–inhibitory population — each checked against the closed forms its own
//! paper derives.
//!
//! # Why two variables
//!
//! The Hodgkin–Huxley membrane has four state variables and cannot be drawn. Both models here keep
//! the qualitative behaviour — a threshold, a resting state that can lose its stability, a limit
//! cycle, hysteresis — in two, and two can be drawn: every question about them becomes a question
//! about where two curves (the NULLCLINES, or isoclines) cross and what the linearisation does there.
//! That is why they are taught, and why every claim below is a closed form rather than a simulation.
//!
//! # `FitzHugh`'s BVP model
//!
//! `FitzHugh`, *Impulses and physiological states in theoretical models of nerve membrane*,
//! Biophysical Journal 1:445–466, 1961, eqs. (1)–(3), in his notation:
//!
//! ```text
//! dx/dt = c (y + x − x³/3 + z)                 (1)
//! dy/dt = −(x − a + b y) / c                   (2)
//! 1 − 2b/3 < a < 1,   0 < b < 1,   b < c²      (3)
//! ```
//!
//! with `a = 0.7`, `b = 0.8`, `c = 3` (his Fig. 1). ⚠ **His `x` runs the other way from a membrane
//! potential**: the resting point sits on the RIGHT branch of the cubic, at `(1.20, −0.625)` in his
//! Fig. 1, and it takes a NEGATIVE stimulus `z` to excite the model. The commonly quoted
//! `v̇ = v − v³/3 − w + I` form is a rescaled, sign-flipped relative; this module keeps `FitzHugh`'s.
//!
//! What he derives, and this module checks:
//!
//! - the nullclines `y = −x + x³/3 − z` (4) and `y = (a − x)/b` (5), which conditions (3) guarantee
//!   cross exactly once — so the singular point is the ONE real root of a monotone cubic, found here
//!   in closed form by Cardano's formula ([`Bvp::singular_point`]);
//! - the linearisation `M = [[(1 − x₁²)c, c], [−1/c, −b/c]]` and its characteristic polynomial
//!   `λ² + [b/c − (1 − x₁²)c]λ + [1 − (1 − x₁²)b] = 0`;
//! - stability iff (7) `b/c − (1 − x₁²)c > 0` and (8) `1 − (1 − x₁²)b > 0`, so the singular point
//!   is unstable exactly for `|x₁| < (1 − b/c²)^½` — [`Bvp::unstable_half_width`];
//! - and therefore, since `z` moves the singular point monotonically along the cubic, an interval
//!   of stimulus inside which the resting state is unstable and the model fires repetitively:
//!   [`Bvp::instability_window`], `(−1.4035, −0.3465)` for his parameters.
//!
//! # Wilson and Cowan's population
//!
//! Wilson and Cowan, *Excitatory and inhibitory interactions in localized populations of model
//! neurons*, Biophysical Journal 12:1–24, 1972, eqs. (11)–(22):
//!
//! ```text
//! τ_e dE/dt = −E + (k_e − r_e E) S_e(c₁E − c₂I + P)          (11)
//! τ_i dI/dt = −I + (k_i − r_i I) S_i(c₃E − c₄I + Q)          (12)
//! S(x) = 1/(1 + exp[−a(x − θ)]) − 1/(1 + exp(aθ))           (15)
//! ```
//!
//! The subtraction in (15) makes `S(0) = 0`, so `E = I = 0` is a steady state with no input, and it
//! lowers the sigmoid's ceiling below one: `k = 1 − 1/(1 + exp(aθ))`, which is the paper's `k_e`
//! and `k_i`, "the maximum values of the response functions". The paper then gives the isoclines by
//! inverting the sigmoid, (13) and (14); the steepest slope `a/4` (16); the condition `c₁ > 9/a_e`
//! (17) for the kink that makes hysteresis possible; `a_e c₂/(a_e c₁ − 9) > (a_i c₄ + 9)/(a_i c₃)`
//! (18) for five steady states; and for a limit cycle, `c₁a_e > c₄a_i + 18` (20) with (21) = (18)
//! and `(a_e c₁ − 9)/(a_e c₂) < 1` (22). Its four parameter sets — Figs. 4, 6, 7/8 and 11 — are
//! [`WilsonCowan::FIG4`] and its siblings, and each figure's count and stability of steady states
//! is checked against what the figure shows.
//!
//! ⚠ **Condition (20) is sufficient and not necessary, and the paper's own limit cycle shows it.**
//! Fig. 11's parameters give `c₁a_e = 20.8` against `c₄a_i + 18 = 24`: (20) FAILS, and the limit
//! cycle is there anyway. The paper says "a sufficient (but not necessary) condition"; a reader
//! using (20) as a test would reject the figure that illustrates it.
//!
//! # Units
//!
//! `FitzHugh`'s time is dimensionless. Wilson–Cowan's `E` and `I` are proportions of a population
//! firing and `P`, `Q` are dimensionless inputs; `τ` is in seconds, as everywhere in this crate.
//!
//! # Not here yet
//!
//! Morris–Lecar (1981) and Hindmarsh–Rose (1984) belong beside these. Neither is here, because this
//! review could not read either paper: the Biophysical Journal full text of Morris and Lecar
//! returned 403, and Hindmarsh and Rose's Proceedings B paper was reachable only through a service
//! that does not serve it. Transcribing their parameter tables from a secondary copy is how this
//! crate once shipped a pre-silicon energy graded as a measurement, so they wait for the papers.

use core::fmt;

/// Why a phase-plane question could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanarError {
    /// `FitzHugh`'s conditions (3) do not hold, so the nullclines need not cross exactly once.
    BvpConditions {
        /// `a`.
        a: f64,
        /// `b`.
        b: f64,
        /// `c`.
        c: f64,
    },
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A state or input that is not a finite number.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// `r_e` outside `(−1/k_e, 1/s₀)`, where the `dE/dt = 0` isocline runs off to infinity and no
    /// finite scan can find every steady state.
    UnboundedIsocline {
        /// `r_e`.
        re: f64,
        /// `−1/k_e`, the lower end of the range that keeps the isocline bounded.
        low: f64,
        /// `1/s₀`, the upper end.
        high: f64,
    },
}

impl fmt::Display for PlanarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BvpConditions { a, b, c } => write!(
                f,
                "a = {a}, b = {b}, c = {c} break FitzHugh's conditions 1 - 2b/3 < a < 1, 0 < b < 1, b < c^2"
            ),
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::UnboundedIsocline { re, low, high } => write!(
                f,
                "r_e = {re} puts the dE/dt = 0 isocline out to infinity; a finite scan needs {low} < r_e < {high}"
            ),
        }
    }
}

impl std::error::Error for PlanarError {}

fn finite(what: &'static str, value: f64) -> Result<f64, PlanarError> {
    if value.is_finite() { Ok(value) } else { Err(PlanarError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, PlanarError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(PlanarError::NotPositive { what, value }) }
}

/// One classical fourth-order Runge–Kutta step of a planar field.
fn rk4(f: impl Fn(f64, f64) -> (f64, f64), x: f64, y: f64, h: f64) -> (f64, f64) {
    let (k1x, k1y) = f(x, y);
    let (k2x, k2y) = f(x + 0.5 * h * k1x, y + 0.5 * h * k1y);
    let (k3x, k3y) = f(x + 0.5 * h * k2x, y + 0.5 * h * k2y);
    let (k4x, k4y) = f(x + h * k3x, y + h * k3y);
    (x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x), y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y))
}

/// A 2×2 real matrix's trace and determinant, which decide a planar fixed point's stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linearisation {
    /// The Jacobian, row-major: `[[∂ẋ/∂x, ∂ẋ/∂y], [∂ẏ/∂x, ∂ẏ/∂y]]`.
    pub jacobian: [[f64; 2]; 2],
}

impl Linearisation {
    /// `λ₁ + λ₂`.
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.jacobian[0][0] + self.jacobian[1][1]
    }

    /// `λ₁ λ₂`.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.jacobian[0][0] * self.jacobian[1][1] - self.jacobian[0][1] * self.jacobian[1][0]
    }

    /// Asymptotically stable: both eigenvalues in the open left half-plane, which for a 2×2 matrix is
    /// exactly `trace < 0` and `determinant > 0`.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.trace() < 0.0 && self.determinant() > 0.0
    }

    /// The two eigenvalues as `(re, im)` pairs, larger real part first.
    #[must_use]
    pub fn eigenvalues(&self) -> [(f64, f64); 2] {
        let (t, d) = (self.trace(), self.determinant());
        let disc = t * t / 4.0 - d;
        if disc >= 0.0 {
            let r = disc.sqrt();
            [(t / 2.0 + r, 0.0), (t / 2.0 - r, 0.0)]
        } else {
            let w = (-disc).sqrt();
            [(t / 2.0, w), (t / 2.0, -w)]
        }
    }
}

/// `FitzHugh`'s Bonhoeffer–van der Pol model, eqs. (1)–(2) of his 1961 paper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bvp {
    /// `a`: where the `y` nullcline crosses the `x` axis. `1 − 2b/3 < a < 1`.
    pub a: f64,
    /// `b`: minus the reciprocal slope of the `y` nullcline. `0 < b < 1`.
    pub b: f64,
    /// `c`: the time-scale separation between the fast `x` and the slow `y`. `b < c²`.
    pub c: f64,
}

impl Bvp {
    /// `FitzHugh`'s own parameters, his Fig. 1: `a = 0.7`, `b = 0.8`, `c = 3`.
    pub const FITZHUGH_1961: Self = Self { a: 0.7, b: 0.8, c: 3.0 };

    /// A model with `FitzHugh`'s conditions (3) checked: `1 − 2b/3 < a < 1`, `0 < b < 1`, `b < c²`.
    ///
    /// # Errors
    ///
    /// [`PlanarError::BvpConditions`] if any of the three fails or any parameter is not finite.
    pub fn new(a: f64, b: f64, c: f64) -> Result<Self, PlanarError> {
        let ok = a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && 1.0 - 2.0 * b / 3.0 < a
            && a < 1.0
            && 0.0 < b
            && b < 1.0
            && b < c * c;
        if ok { Ok(Self { a, b, c }) } else { Err(PlanarError::BvpConditions { a, b, c }) }
    }

    /// The vector field `(dx/dt, dy/dt)` at `(x, y)` under stimulus `z`, eqs. (1) and (2).
    #[must_use]
    pub fn field(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        (self.c * (y + x - x * x * x / 3.0 + z), -(x - self.a + self.b * y) / self.c)
    }

    /// The singular point `(x₁, y₁)` under stimulus `z`: where nullclines (4) and (5) cross.
    ///
    /// Eliminating `y` gives `x³/3 + (1/b − 1)x − (a/b + z) = 0`. With `0 < b < 1` the linear
    /// coefficient is positive, so the cubic is strictly increasing and has exactly one real root —
    /// `FitzHugh`'s "only one intersection" — which Cardano's formula gives in closed form:
    /// `x³ + px + q = 0` with `p = 3(1/b − 1) > 0` has the single real root
    /// `∛(−q/2 + √Δ) + ∛(−q/2 − √Δ)`, `Δ = q²/4 + p³/27 > 0`.
    #[must_use]
    pub fn singular_point(&self, z: f64) -> (f64, f64) {
        let p = 3.0 * (1.0 / self.b - 1.0);
        let q = -3.0 * (self.a / self.b + z);
        let delta = q * q / 4.0 + p * p * p / 27.0;
        let s = delta.sqrt();
        let x = (-q / 2.0 + s).cbrt() + (-q / 2.0 - s).cbrt();
        (x, (self.a - x) / self.b)
    }

    /// The linearisation at a point on the cubic with abscissa `x1`: `FitzHugh`'s matrix
    /// `M = [[(1 − x₁²)c, c], [−1/c, −b/c]]`.
    #[must_use]
    pub fn linearisation(&self, x1: f64) -> Linearisation {
        let c = self.c;
        Linearisation { jacobian: [[(1.0 - x1 * x1) * c, c], [-1.0 / c, -self.b / c]] }
    }

    /// `FitzHugh`'s stability conditions (7) and (8) at `x1`, both required.
    #[must_use]
    pub fn is_stable(&self, x1: f64) -> bool {
        let g = 1.0 - x1 * x1;
        self.b / self.c - g * self.c > 0.0 && 1.0 - g * self.b > 0.0
    }

    /// `(1 − b/c²)^½`: the singular point is unstable exactly when `|x₁|` is below this.
    ///
    /// From (9), `1 − x₁² < b/c²` is the stability condition that can fail; (10) cannot, because
    /// `b < 1` makes `1/b > 1`.
    #[must_use]
    pub fn unstable_half_width(&self) -> f64 {
        (1.0 - self.b / (self.c * self.c)).sqrt()
    }

    /// The stimulus that puts the singular point at abscissa `x1`, from nullclines (4) and (5):
    /// `z = x₁³/3 + (1/b − 1)x₁ − a/b`. Strictly increasing in `x₁`.
    #[must_use]
    pub fn stimulus_at(&self, x1: f64) -> f64 {
        x1 * x1 * x1 / 3.0 + (1.0 / self.b - 1.0) * x1 - self.a / self.b
    }

    /// The open interval of stimulus `z` inside which the resting state is unstable.
    ///
    /// Its ends are [`Bvp::stimulus_at`] the two edges `±`[`Bvp::unstable_half_width`], in order;
    /// for `FitzHugh`'s parameters, about `(−1.4035, −0.3465)`.
    #[must_use]
    pub fn instability_window(&self) -> (f64, f64) {
        let w = self.unstable_half_width();
        (self.stimulus_at(-w), self.stimulus_at(w))
    }

    /// One fourth-order Runge–Kutta step of length `h` from `(x, y)` under constant `z`.
    ///
    /// The parameters are checked here as well as in [`Bvp::new`], because the fields are public: a
    /// model built outside conditions (3) is still an ODE worth integrating — it has more than one
    /// singular point, which is a lesson in itself — but one with a non-finite parameter or a
    /// non-positive `c` is not, and would return a NaN state as if it were an answer.
    ///
    /// # Errors
    ///
    /// [`PlanarError::NotPositive`] for a step or a `c` that is not finite and positive;
    /// [`PlanarError::NonFinite`] for a parameter, state or stimulus that is not finite.
    pub fn step(&self, x: f64, y: f64, z: f64, h: f64) -> Result<(f64, f64), PlanarError> {
        finite("a", self.a)?;
        finite("b", self.b)?;
        positive("c", self.c)?;
        positive("h", h)?;
        finite("x", x)?;
        finite("y", y)?;
        finite("z", z)?;
        Ok(rk4(|x, y| self.field(x, y, z), x, y, h))
    }
}

/// Wilson and Cowan's sigmoid, their eq. (15): a logistic shifted down so that `S(0) = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sigmoid {
    /// `a`: sets the maximum slope, `a/4` (their eq. 16).
    pub a: f64,
    /// `θ`: where that maximum slope is.
    pub theta: f64,
}

impl Sigmoid {
    /// `1/(1 + exp(aθ))`: the logistic's value at zero, which eq. (15) subtracts.
    #[must_use]
    pub fn offset(&self) -> f64 {
        1.0 / (1.0 + (self.a * self.theta).exp())
    }

    /// `S(x) = 1/(1 + exp[−a(x − θ)]) − 1/(1 + exp(aθ))`.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.a * (x - self.theta)).exp()) - self.offset()
    }

    /// `S'(x) = a σ (1 − σ)` with `σ` the unshifted logistic; its maximum is `a/4`, at `θ`.
    #[must_use]
    pub fn slope(&self, x: f64) -> f64 {
        let s = 1.0 / (1.0 + (-self.a * (x - self.theta)).exp());
        self.a * s * (1.0 - s)
    }

    /// The ceiling, `S(∞) = 1 − 1/(1 + exp(aθ))`: the paper's `k`, "the maximum values of the
    /// response functions".
    #[must_use]
    pub fn ceiling(&self) -> f64 {
        1.0 - self.offset()
    }

    /// `S⁻¹(y) = θ − ln(1/(y + s₀) − 1)/a`, defined for `−s₀ < y < k`; `None` outside it.
    #[must_use]
    pub fn inverse(&self, y: f64) -> Option<f64> {
        let u = y + self.offset();
        if u > 0.0 && u < 1.0 { Some(self.theta - (1.0 / u - 1.0).ln() / self.a) } else { None }
    }
}

/// Wilson and Cowan's excitatory–inhibitory population, their eqs. (11) and (12).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonCowan {
    /// Excitatory-to-excitatory coupling.
    pub c1: f64,
    /// Inhibitory-to-excitatory coupling (enters with a minus sign).
    pub c2: f64,
    /// Excitatory-to-inhibitory coupling.
    pub c3: f64,
    /// Inhibitory-to-inhibitory coupling (enters with a minus sign).
    pub c4: f64,
    /// The excitatory response function `S_e`.
    pub se: Sigmoid,
    /// The inhibitory response function `S_i`.
    pub si: Sigmoid,
    /// The excitatory refractory factor `r_e`. The paper sets it to 1 "from now on" after eq. (17).
    pub re: f64,
    /// The inhibitory refractory factor `r_i`, likewise 1.
    pub ri: f64,
    /// The excitatory time constant `τ_e`, seconds.
    pub tau_e: f64,
    /// The inhibitory time constant `τ_i`, seconds.
    pub tau_i: f64,
}

/// A steady state of [`WilsonCowan`] and whether it is stable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyState {
    /// Excitatory activity `E`.
    pub e: f64,
    /// Inhibitory activity `I`.
    pub i: f64,
    /// Whether the linearisation there is asymptotically stable.
    pub stable: bool,
}

impl WilsonCowan {
    /// Fig. 4: three steady states at `P = Q = 0`, two stable and one unstable.
    ///
    /// The paper gives no `τ` for this figure; steady states and their stability do not depend on a
    /// common `τ`, and 10 ms is the value it adopts on p. 17.
    pub const FIG4: Self = Self {
        c1: 12.0,
        c2: 4.0,
        c3: 13.0,
        c4: 11.0,
        se: Sigmoid { a: 1.2, theta: 2.8 },
        si: Sigmoid { a: 1.0, theta: 4.0 },
        re: 1.0,
        ri: 1.0,
        tau_e: 10e-3,
        tau_i: 10e-3,
    };

    /// Fig. 6: two separated hysteresis loops as `P` varies.
    pub const FIG6: Self = Self {
        c1: 13.0,
        c2: 4.0,
        c3: 20.0,
        c4: 2.0,
        se: Sigmoid { a: 1.2, theta: 2.7 },
        si: Sigmoid { a: 5.0, theta: 3.7 },
        re: 1.0,
        ri: 1.0,
        tau_e: 10e-3,
        tau_i: 10e-3,
    };

    /// Figs. 7 and 8: five steady states at `P = 0`, three stable and two unstable.
    pub const FIG7: Self = Self {
        c1: 13.0,
        c2: 4.0,
        c3: 22.0,
        c4: 2.0,
        se: Sigmoid { a: 1.5, theta: 2.5 },
        si: Sigmoid { a: 6.0, theta: 4.3 },
        re: 1.0,
        ri: 1.0,
        tau_e: 10e-3,
        tau_i: 10e-3,
    };

    /// Fig. 11: a limit cycle under constant stimulation `P = 1.25`, with `τ = 8` ms.
    pub const FIG11: Self = Self {
        c1: 16.0,
        c2: 12.0,
        c3: 15.0,
        c4: 3.0,
        se: Sigmoid { a: 1.3, theta: 4.0 },
        si: Sigmoid { a: 2.0, theta: 3.7 },
        re: 1.0,
        ri: 1.0,
        tau_e: 8e-3,
        tau_i: 8e-3,
    };

    /// The stimulus Fig. 11's limit cycle is drawn at.
    pub const FIG11_P: f64 = 1.25;

    /// The vector field `(dE/dt, dI/dt)` at `(E, I)` under inputs `P` and `Q`, eqs. (11) and (12).
    #[must_use]
    pub fn field(&self, e: f64, i: f64, p: f64, q: f64) -> (f64, f64) {
        let (ke, ki) = (self.se.ceiling(), self.si.ceiling());
        let de = (-e + (ke - self.re * e) * self.se.eval(self.c1 * e - self.c2 * i + p)) / self.tau_e;
        let di = (-i + (ki - self.ri * i) * self.si.eval(self.c3 * e - self.c4 * i + q)) / self.tau_i;
        (de, di)
    }

    /// The linearisation of [`WilsonCowan::field`] at `(E, I)`.
    #[must_use]
    pub fn linearisation(&self, e: f64, i: f64, p: f64, q: f64) -> Linearisation {
        let (ke, ki) = (self.se.ceiling(), self.si.ceiling());
        let u = self.c1 * e - self.c2 * i + p;
        let v = self.c3 * e - self.c4 * i + q;
        let (su, dsu) = (self.se.eval(u), self.se.slope(u));
        let (sv, dsv) = (self.si.eval(v), self.si.slope(v));
        let ae = ke - self.re * e;
        let ai = ki - self.ri * i;
        Linearisation {
            jacobian: [
                [(-1.0 - self.re * su + ae * dsu * self.c1) / self.tau_e, (-ae * dsu * self.c2) / self.tau_e],
                [(ai * dsv * self.c3) / self.tau_i, (-1.0 - self.ri * sv - ai * dsv * self.c4) / self.tau_i],
            ],
        }
    }

    /// Every parameter finite, and the ones that must be positive positive: the couplings, both
    /// sigmoid slopes and both time constants. The thresholds and refractory factors may take any
    /// finite value.
    ///
    /// # Errors
    ///
    /// [`PlanarError::NotPositive`] or [`PlanarError::NonFinite`], naming the first parameter that
    /// fails.
    pub fn check(&self) -> Result<(), PlanarError> {
        for (what, value) in [
            ("c1", self.c1),
            ("c2", self.c2),
            ("c3", self.c3),
            ("c4", self.c4),
            ("a_e", self.se.a),
            ("a_i", self.si.a),
            ("tau_e", self.tau_e),
            ("tau_i", self.tau_i),
        ] {
            positive(what, value)?;
        }
        for (what, value) in [("theta_e", self.se.theta), ("theta_i", self.si.theta), ("r_e", self.re), ("r_i", self.ri)] {
            finite(what, value)?;
        }
        Ok(())
    }

    /// The `dE/dt = 0` isocline, eq. (13): the `I` it passes through at a given `E`.
    ///
    /// `c₂I = c₁E − S_e⁻¹(E/(k_e − r_e E)) + P`, and `None` exactly where `E/(k_e − r_e E)` is
    /// outside the range of `S_e`.
    ///
    /// ⚠ That includes points past the pole `E = k_e/r_e`, where `k_e − r_e E` is negative. The
    /// first version refused every such point as "past the ceiling". For `r_e` inside
    /// `(−1/k_e, 1/s₀)` that made no difference, because there the far branch of `E/(k_e − r_e E)`
    /// lies wholly outside the sigmoid's range ([`WilsonCowan::e_isocline_domain`]). Outside it, it
    /// does not: with `r_e = 40` and Fig. 4's sigmoid, `E = 0.2` gives `E/(k_e − r_e E) = −0.028`,
    /// above the floor `−s₀ = −0.034`, and `E = (k_e − r_e E)S_e(·)` holds with both factors
    /// negative — a real point of the isocline, at which `dE/dt` vanishes, and the guard discarded it.
    #[must_use]
    pub fn e_isocline(&self, e: f64, p: f64) -> Option<f64> {
        let inv = self.se.inverse(e / (self.se.ceiling() - self.re * e))?;
        Some((self.c1 * e - inv + p) / self.c2)
    }

    /// The `dI/dt = 0` isocline, eq. (14): the `E` it passes through at a given `I`.
    ///
    /// `c₃E = c₄I + S_i⁻¹(I/(k_i − r_i I)) − Q`, and `None` exactly where `I/(k_i − r_i I)` is outside
    /// the range of `S_i` — past the pole included, as for [`WilsonCowan::e_isocline`].
    #[must_use]
    pub fn i_isocline(&self, i: f64, q: f64) -> Option<f64> {
        let inv = self.si.inverse(i / (self.si.ceiling() - self.ri * i))?;
        Some((self.c4 * i + inv - q) / self.c3)
    }

    /// The interval of `E` on which the `dE/dt = 0` isocline exists, when it is bounded.
    ///
    /// Solving `E/(k − rE) = y` gives `E = yk/(1 + ry)`, increasing in `y`, so the sigmoid's range
    /// `−s₀ < y < k` maps onto `(−s₀k/(1 − rs₀), k²/(1 + rk))`. Both ends are finite exactly when
    /// both denominators are positive, `−1/k < r < 1/s₀`, and on that range the other branch of the
    /// hyperbola, past the pole, lies wholly outside the sigmoid's range — so this interval is ALL of
    /// the isocline. The paper's `r_e = 1` is inside it for every sigmoid with `s₀ < 1`.
    ///
    /// # Errors
    ///
    /// [`PlanarError::UnboundedIsocline`] for an `r_e` outside `(−1/k_e, 1/s₀)`, where the isocline
    /// runs to infinity in one direction.
    pub fn e_isocline_domain(&self) -> Result<(f64, f64), PlanarError> {
        let k = self.se.ceiling();
        let s0 = self.se.offset();
        if 1.0 + self.re * k > 0.0 && 1.0 - self.re * s0 > 0.0 {
            Ok((-s0 * k / (1.0 - self.re * s0), k * k / (1.0 + self.re * k)))
        } else {
            Err(PlanarError::UnboundedIsocline { re: self.re, low: -1.0 / k, high: 1.0 / s0 })
        }
    }

    /// Every steady state, found as the crossings of the two isoclines, each with its stability.
    ///
    /// Walks `E` across the whole interval on which isocline (13) exists —
    /// [`WilsonCowan::e_isocline_domain`] — on a grid of `samples` points, and evaluates `dI/dt` ALONG that isocline:
    /// a steady state is exactly a point of the `dE/dt = 0` curve where `dI/dt` vanishes too, and each
    /// transversal crossing of the other isocline changes its sign. Every sign change is bisected to
    /// the last bit. Two crossings closer than one grid step would be missed; the tests use the
    /// paper's figures, whose steady states are well separated, and check the COUNT against each.
    ///
    /// ⚠ The obvious alternative — comparing the two isoclines by inverting `S_i` as eq. (14) does —
    /// was this function's first version, and it found ONE of Fig. 7's five steady states. With
    /// `a_i = 6` the inverse exists only for `I > −6 × 10⁻¹²`, so wherever isocline (13) dips below
    /// zero or rises past `S_i`'s range the comparison was undefined, and every crossing at the edge of
    /// that domain — the origin among them — lost its bracket. The field itself is defined for every
    /// real `(E, I)`.
    ///
    /// # Errors
    ///
    /// [`PlanarError::NotPositive`] for fewer than two samples (reported as the count);
    /// [`PlanarError::NonFinite`] for an input that is not finite; whatever [`WilsonCowan::check`]
    /// and [`WilsonCowan::e_isocline_domain`] refuse.
    pub fn steady_states(&self, p: f64, q: f64, samples: usize) -> Result<Vec<SteadyState>, PlanarError> {
        self.check()?;
        finite("P", p)?;
        finite("Q", q)?;
        if samples < 2 {
            return Err(PlanarError::NotPositive { what: "samples", value: samples as f64 });
        }
        let (lo, hi) = self.e_isocline_domain()?;
        let ki = self.si.ceiling();
        let mismatch = |e: f64| -> Option<f64> {
            let i = self.e_isocline(e, p)?;
            Some(-i + (ki - self.ri * i) * self.si.eval(self.c3 * e - self.c4 * i + q))
        };
        let step = (hi - lo) / samples as f64;
        // Every grid point is strictly inside the domain, where the isocline exists; `filter_map`
        // drops a point only if rounding at an end of the domain puts it outside.
        let grid: Vec<(f64, f64)> = (1..samples)
            .filter_map(|n| {
                let e = lo + step * n as f64;
                mismatch(e).map(|g| (e, g))
            })
            .collect();
        let mut out = Vec::new();
        for pair in grid.windows(2) {
            let [(e0, g0), (e, g)] = [pair[0], pair[1]];
            if (g0 < 0.0) != (g < 0.0) {
                let (mut a, mut b, mut ga) = (e0, e, g0);
                for _ in 0..200 {
                    let m = 0.5 * (a + b);
                    let Some(gm) = mismatch(m) else { break };
                    if (gm < 0.0) == (ga < 0.0) {
                        a = m;
                        ga = gm;
                    } else {
                        b = m;
                    }
                }
                let e_star = 0.5 * (a + b);
                if let Some(i_star) = self.e_isocline(e_star, p) {
                    let stable = self.linearisation(e_star, i_star, p, q).is_stable();
                    out.push(SteadyState { e: e_star, i: i_star, stable });
                }
            }
        }
        Ok(out)
    }

    /// Condition (17), `c₁ > 9/a_e`: sufficient for the kink in isocline (13) that hysteresis needs.
    #[must_use]
    pub fn kink_condition(&self) -> bool {
        self.c1 > 9.0 / self.se.a
    }

    /// Condition (18) = (21), `a_e c₂/(a_e c₁ − 9) > (a_i c₄ + 9)/(a_i c₃)`: sufficient for five steady
    /// states under some constant inputs. Requires `a_e c₁ > 9`, as the paper notes.
    #[must_use]
    pub fn five_state_condition(&self) -> bool {
        let (ae, ai) = (self.se.a, self.si.a);
        ae * self.c1 > 9.0 && ae * self.c2 / (ae * self.c1 - 9.0) > (ai * self.c4 + 9.0) / (ai * self.c3)
    }

    /// Condition (20), `c₁a_e > c₄a_i + 18`: sufficient — and NOT necessary — for the single steady
    /// state near the inflection points to be unstable.
    #[must_use]
    pub fn instability_condition(&self) -> bool {
        self.c1 * self.se.a > self.c4 * self.si.a + 18.0
    }

    /// Condition (22), `(a_e c₁ − 9)/(a_e c₂) < 1`: one steady state rather than five.
    #[must_use]
    pub fn single_state_condition(&self) -> bool {
        (self.se.a * self.c1 - 9.0) / (self.se.a * self.c2) < 1.0
    }

    /// One fourth-order Runge–Kutta step of length `h` seconds.
    ///
    /// # Errors
    ///
    /// [`PlanarError::NotPositive`] for a step that is not finite and positive;
    /// [`PlanarError::NonFinite`] for a state or input that is not finite; whatever
    /// [`WilsonCowan::check`] refuses.
    pub fn step(&self, e: f64, i: f64, p: f64, q: f64, h: f64) -> Result<(f64, f64), PlanarError> {
        self.check()?;
        positive("h", h)?;
        finite("E", e)?;
        finite("I", i)?;
        finite("P", p)?;
        finite("Q", q)?;
        Ok(rk4(|e, i| self.field(e, i, p, q), e, i, h))
    }
}

#[cfg(test)]
mod tests {
    use super::{Bvp, Linearisation, PlanarError, Sigmoid, WilsonCowan};

    /// `FitzHugh`'s resting point is where his Fig. 1 prints it, and the printed `y` is computed from
    /// the rounded `x`.
    ///
    /// At `z = 0` the singular point is `x₁ = 1.199408…`, which he prints as `1.20`. The `y` he prints,
    /// `−0.625`, is `(a − 1.20)/b` EXACTLY — the nullcline evaluated at the rounded abscissa. The true
    /// ordinate is `−0.624260…`. Both are pinned so the difference is on the page and nobody "corrects"
    /// the model to match the printed digit.
    #[test]
    fn fitzhughs_resting_point_is_where_his_figure_prints_it() {
        let m = Bvp::FITZHUGH_1961;
        let (x, y) = m.singular_point(0.0);
        assert_eq!((x * 100.0).round(), 120.0, "x = {x}");
        assert!((y - (-0.624_260_044)).abs() < 1e-9, "y = {y}");
        assert_eq!((m.a - 1.20) / m.b, -0.625, "the printed −0.625, from x = 1.20");
        // And it is on both nullclines, eqs. (4) and (5), to rounding.
        let (dx, dy) = m.field(x, y, 0.0);
        assert!(dx.abs() < 1e-14 && dy.abs() < 1e-15, "{dx} {dy}");
    }

    /// Cardano's root is THE root of the singular-point cubic, for every stimulus.
    ///
    /// `x³/3 + (1/b − 1)x − (a/b + z)` is strictly increasing when `b < 1`, so a residual at rounding
    /// level is a proof of uniqueness as well as of correctness. Swept across a range of `z` that
    /// crosses both ends of the instability window.
    #[test]
    fn cardanos_root_is_the_one_root_of_the_singular_point_cubic() {
        let m = Bvp::FITZHUGH_1961;
        for n in -300..=300 {
            let z = f64::from(n) / 100.0;
            let (x, y) = m.singular_point(z);
            let r = x * x * x / 3.0 + (1.0 / m.b - 1.0) * x - (m.a / m.b + z);
            assert!(r.abs() < 1e-13, "z = {z}: residual {r}");
            assert!((y - (m.a - x) / m.b).abs() < 1e-15);
            assert!((m.stimulus_at(x) - z).abs() < 1e-13, "stimulus_at is the inverse");
        }
    }

    /// `FitzHugh`'s characteristic polynomial is the one of his matrix `M`, and (7)–(8) are the
    /// Routh–Hurwitz conditions on it.
    ///
    /// `λ² + [b/c − (1 − x₁²)c]λ + [1 − (1 − x₁²)b]`: its linear coefficient is minus the trace of
    /// `M` and its constant term is the determinant. Both are checked exactly at a spread of `x₁`,
    /// and the stability verdict from (7)–(8) is checked against the trace–determinant verdict.
    #[test]
    fn his_polynomial_is_the_matrix_and_his_conditions_are_its_stability() {
        let m = Bvp::FITZHUGH_1961;
        for n in -250..=250 {
            let x1 = f64::from(n) / 100.0;
            let l = m.linearisation(x1);
            let g = 1.0 - x1 * x1;
            assert!((-l.trace() - (m.b / m.c - g * m.c)).abs() < 1e-14);
            assert!((l.determinant() - (1.0 - g * m.b)).abs() < 1e-14);
            assert_eq!(m.is_stable(x1), l.is_stable(), "x1 = {x1}");
        }
    }

    /// The resting state is unstable exactly for `|x₁| < (1 − b/c²)^½`, and at the edge the
    /// eigenvalues are purely imaginary.
    ///
    /// `(1 − 0.8/9)^½ = 0.954521…` — measured `0.954 521 404 2`. Just inside the edge the singular point is unstable, just outside
    /// it is stable, and AT the edge the trace vanishes while the determinant stays positive — the
    /// Andronov–Hopf signature: `λ = ±iω` with `ω² = det`.
    #[test]
    fn the_resting_state_loses_stability_exactly_at_the_half_width() {
        let m = Bvp::FITZHUGH_1961;
        let w = m.unstable_half_width();
        assert!((w - 0.954_521_404_2).abs() < 1e-10, "{w}");
        for s in [-1.0, 1.0] {
            assert!(!m.is_stable(s * (w - 1e-6)), "just inside");
            assert!(m.is_stable(s * (w + 1e-6)), "just outside");
            let l = m.linearisation(s * w);
            assert!(l.trace().abs() < 1e-14, "the trace vanishes: {}", l.trace());
            let [(re, im), _] = l.eigenvalues();
            assert!(re.abs() < 1e-14 && (im * im - l.determinant()).abs() < 1e-14);
            assert!(l.determinant() > 0.0);
        }
    }

    /// The stimulus window inside which the model fires repetitively, and the model firing there.
    ///
    /// The window's ends are `stimulus_at(∓w)`: measured `−1.403 522 0…` and `−0.346 478 0…` for `FitzHugh`'s
    /// parameters. Integrated from a point near the singular point, a stimulus inside the window
    /// grows into a large limit cycle — `x` swinging across both outer branches of the cubic — and one
    /// outside it settles back to rest. The swing is the observable a phase-plane teacher would
    /// point at, and it is a factor of hundreds between the two cases.
    #[test]
    fn inside_the_stimulus_window_the_model_fires_and_outside_it_rests() {
        let m = Bvp::FITZHUGH_1961;
        let (lo, hi) = m.instability_window();
        assert!((lo - (-1.403_522_037)).abs() < 1e-8 && (hi - (-0.346_477_963)).abs() < 1e-8, "({lo}, {hi})");
        let swing = |z: f64| {
            let (x0, y0) = m.singular_point(z);
            let (mut x, mut y) = (x0 + 1e-3, y0);
            for _ in 0..40_000 {
                (x, y) = m.step(x, y, z, 5e-3).unwrap();
            }
            let (mut xmin, mut xmax) = (x, x);
            for _ in 0..20_000 {
                (x, y) = m.step(x, y, z, 5e-3).unwrap();
                xmin = xmin.min(x);
                xmax = xmax.max(x);
            }
            xmax - xmin
        };
        for z in [-1.2, -0.9, -0.5] {
            assert!(swing(z) > 2.5, "z = {z} is inside the window: {}", swing(z));
        }
        for z in [0.0, -0.2, -1.6] {
            assert!(swing(z) < 1e-6, "z = {z} is outside the window: {}", swing(z));
        }
    }

    /// `FitzHugh`'s conditions (3) are enforced, each at its own boundary.
    #[test]
    fn fitzhughs_parameter_conditions_are_enforced() {
        assert!(Bvp::new(0.7, 0.8, 3.0).is_ok());
        let refused = [
            (1.0, 0.8, 3.0),   // a < 1 fails
            (0.4, 0.8, 3.0),   // 1 − 2b/3 = 0.4667 < a fails
            (0.7, 1.0, 3.0),   // b < 1 fails
            (0.7, 0.0, 3.0),   // 0 < b fails
            (0.7, 0.8, 0.8),   // b < c² = 0.64 fails
            (f64::NAN, 0.8, 3.0),
        ];
        for (a, b, c) in refused {
            assert!(matches!(Bvp::new(a, b, c), Err(PlanarError::BvpConditions { .. })), "({a}, {b}, {c})");
        }
        assert_eq!(
            Bvp::new(1.0, 0.8, 3.0).unwrap_err().to_string(),
            "a = 1, b = 0.8, c = 3 break FitzHugh's conditions 1 - 2b/3 < a < 1, 0 < b < 1, b < c^2"
        );
        let m = Bvp::FITZHUGH_1961;
        assert_eq!(m.step(0.0, 0.0, 0.0, 0.0), Err(PlanarError::NotPositive { what: "h", value: 0.0 }));
        assert_eq!(m.step(f64::NAN, 0.0, 0.0, 0.1).unwrap_err().to_string(), "x = NaN is not finite");
    }

    /// Wilson and Cowan's sigmoid: zero at zero, steepest slope `a/4` at `θ`, and an exact inverse.
    ///
    /// `S(0) = 0` holds to the last bit, because `−a(0 − θ)` and `aθ` are the same number in floating
    /// point and the two terms of eq. (15) are then identical. The ceiling is `1 − 1/(1 + exp(aθ))`,
    /// below one as the paper says, and the inverse undoes `S` across its range.
    #[test]
    fn the_sigmoid_is_zero_at_zero_steepest_at_theta_and_invertible() {
        for s in [Sigmoid { a: 1.2, theta: 2.8 }, Sigmoid { a: 6.0, theta: 4.3 }, Sigmoid { a: 1.0, theta: 4.0 }] {
            assert_eq!(s.eval(0.0), 0.0);
            assert_eq!(s.slope(s.theta), s.a / 4.0);
            for n in -40..=40 {
                let x = s.theta + f64::from(n) / 10.0;
                assert!(s.slope(x) <= s.a / 4.0);
                let back = s.inverse(s.eval(x)).unwrap();
                // The inverse is ill-conditioned in the tails, and the bound says how much: an error
                // of a few units in the last place of `S(x)` moves the preimage by that error over
                // `S'(x)`. At `a = 6`, `x − θ = 3.1` the sigmoid is flat to 8 × 10⁻⁹ and the round
                // trip measures 2.0 × 10⁻⁹ — inside `8ε/S'`, and far outside any fixed 10⁻⁹.
                assert!((back - x).abs() <= 8.0 * f64::EPSILON / s.slope(x), "{x} -> {back}");
            }
            assert!((s.ceiling() - (1.0 - 1.0 / (1.0 + (s.a * s.theta).exp()))).abs() < 1e-16);
            assert!(s.ceiling() < 1.0);
            assert_eq!(s.inverse(s.ceiling() + 1e-9), None, "above the ceiling there is no preimage");
            assert_eq!(s.inverse(-s.offset() - 1e-9), None, "nor below the floor");
        }
    }

    /// Fig. 4: three steady states at no input, stable–unstable–stable, and the rest state is one.
    ///
    /// The figure marks the lower and upper intersections `(+)` and the middle one `(−)`. Condition
    /// (17) holds, `12 > 9/1.2 = 7.5`, as the kink needs. `E = I = 0` is a steady state exactly,
    /// because `S(0) = 0`.
    #[test]
    fn figure_4_has_three_steady_states_stable_unstable_stable() {
        let m = WilsonCowan::FIG4;
        assert!(m.kink_condition());
        assert_eq!(m.field(0.0, 0.0, 0.0, 0.0), (0.0, 0.0));
        let ss = m.steady_states(0.0, 0.0, 20_000).unwrap();
        assert_eq!(ss.len(), 3, "{ss:?}");
        assert_eq!(ss.iter().map(|s| s.stable).collect::<Vec<_>>(), vec![true, false, true]);
        assert!(ss[0].e.abs() < 1e-9 && ss[0].i.abs() < 1e-9, "the rest state is the origin: {:?}", ss[0]);
        for s in &ss {
            let (de, di) = m.field(s.e, s.i, 0.0, 0.0);
            assert!(de.abs() < 1e-9 && di.abs() < 1e-9, "{s:?}: {de} {di}");
        }
    }

    /// Figs. 7 and 8: five steady states at `P = 0`, three stable and two unstable, as condition (18)
    /// promises.
    #[test]
    fn figures_7_and_8_have_five_steady_states_three_of_them_stable() {
        let m = WilsonCowan::FIG7;
        assert!(m.five_state_condition());
        let ss = m.steady_states(0.0, 0.0, 40_000).unwrap();
        assert_eq!(ss.len(), 5, "{ss:?}");
        assert_eq!(
            ss.iter().map(|s| s.stable).collect::<Vec<_>>(),
            vec![true, false, true, false, true],
            "alternating, as Fig. 8 marks them"
        );
    }

    /// Fig. 11: one steady state at `P = 1.25`, unstable, and a limit cycle around it — although the
    /// paper's own sufficient condition (20) FAILS for these parameters.
    ///
    /// `c₁a_e = 20.8` against `c₄a_i + 18 = 24`. Conditions (21) and (22) hold. The single steady state
    /// is unstable, and a trajectory started beside it swings `E` over a range that Fig. 11b draws as
    /// roughly 0.05 to 0.3 — checked here as a swing of more than 0.1, against the millionths a
    /// damped approach would leave.
    #[test]
    fn figure_11_cycles_although_the_papers_sufficient_condition_fails() {
        let m = WilsonCowan::FIG11;
        let p = WilsonCowan::FIG11_P;
        assert!(!m.instability_condition(), "20.8 > 24 is false");
        assert!((m.c1 * m.se.a - 20.8).abs() < 1e-12 && m.c4 * m.si.a + 18.0 == 24.0);
        assert!(m.five_state_condition() && m.single_state_condition(), "(21) and (22) hold");
        let ss = m.steady_states(p, 0.0, 20_000).unwrap();
        assert_eq!(ss.len(), 1, "{ss:?}");
        assert!(!ss[0].stable, "the one steady state is unstable");
        let (mut e, mut i) = (ss[0].e + 1e-3, ss[0].i);
        for _ in 0..20_000 {
            (e, i) = m.step(e, i, p, 0.0, 1e-5).unwrap();
        }
        let (mut lo, mut hi) = (e, e);
        for _ in 0..20_000 {
            (e, i) = m.step(e, i, p, 0.0, 1e-5).unwrap();
            lo = lo.min(e);
            hi = hi.max(e);
        }
        assert!(hi - lo > 0.1, "E swings from {lo} to {hi}");
    }

    /// Fig. 6's parameters satisfy (17) and (18), and the conditions refuse what they should.
    #[test]
    fn the_papers_conditions_classify_its_own_figures() {
        let f6 = WilsonCowan::FIG6;
        assert!(f6.kink_condition() && f6.five_state_condition());
        // A population with weak self-excitation has no kink: c₁ = 5 < 9/1.2.
        let weak = WilsonCowan { c1: 5.0, ..WilsonCowan::FIG4 };
        assert!(!weak.kink_condition() && !weak.five_state_condition());
        assert_eq!(
            WilsonCowan::FIG4.steady_states(0.0, 0.0, 1),
            Err(PlanarError::NotPositive { what: "samples", value: 1.0 })
        );
        assert_eq!(
            WilsonCowan::FIG4.step(0.0, 0.0, f64::INFINITY, 0.0, 1e-4),
            Err(PlanarError::NonFinite { what: "P", value: f64::INFINITY })
        );
    }

    /// A linearisation's eigenvalues are its trace and determinant, real or complex.
    #[test]
    fn eigenvalues_are_the_trace_and_the_determinant() {
        let real = Linearisation { jacobian: [[-3.0, 1.0], [0.0, -2.0]] };
        assert_eq!(real.eigenvalues(), [(-2.0, 0.0), (-3.0, 0.0)]);
        let spiral = Linearisation { jacobian: [[-1.0, -4.0], [1.0, -1.0]] };
        assert_eq!(spiral.eigenvalues(), [(-1.0, 2.0), (-1.0, -2.0)]);
        assert!(real.is_stable() && spiral.is_stable());
        let saddle = Linearisation { jacobian: [[1.0, 0.0], [0.0, -1.0]] };
        assert!(!saddle.is_stable());
    }

    /// Both isoclines, eqs. (13) and (14), pass through every steady state.
    ///
    /// The steady-state search walks isocline (13) and never evaluates (14), so (14) is checked here on
    /// its own: at each of Fig. 4's three states, the `E` it returns for the state's `I` is the state's
    /// `E`. The `I` from (13) is the very value the search stored, so that comparison is exact.
    #[test]
    fn both_isoclines_pass_through_every_steady_state() {
        let m = WilsonCowan::FIG4;
        for s in m.steady_states(0.0, 0.0, 20_000).unwrap() {
            assert_eq!(m.e_isocline(s.e, 0.0), Some(s.i));
            assert!((m.i_isocline(s.i, 0.0).unwrap() - s.e).abs() < 1e-9, "{s:?}");
        }
        assert_eq!(m.e_isocline(10.0, 0.0), None, "past the sigmoid's ceiling there is no isocline");
        assert_eq!(m.i_isocline(10.0, 0.0), None);
    }

    /// The shared integrator is fourth order: halving the step divides the error by sixteen.
    ///
    /// `FitzHugh`'s model from `(0.5, 0.2)` under `z = −0.9`, integrated to `t = 2` with steps of
    /// 0.05, 0.025 and 0.0125 and compared with a run at 1/12 800. The measured ratios are 15.95 and
    /// 15.95; a scheme that takes its first midpoint slope a full step out — the one-character slip
    /// the classical tableau invites — measures 1.9, first order.
    #[test]
    fn the_integrator_is_fourth_order() {
        let m = Bvp::FITZHUGH_1961;
        let run = |h: f64, n: usize| {
            let (mut x, mut y) = (0.5, 0.2);
            for _ in 0..n {
                (x, y) = m.step(x, y, -0.9, h).unwrap();
            }
            (x, y)
        };
        let (xr, yr) = run(1.0 / 12_800.0, 25_600);
        let err: Vec<f64> =
            [(0.05, 40), (0.025, 80), (0.0125, 160)].iter().map(|&(h, n)| { let (x, y) = run(h, n); (x - xr).hypot(y - yr) }).collect();
        for pair in err.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((15.0..17.0).contains(&ratio), "{err:?}: ratio {ratio}");
        }
    }

    /// Both linearisations are the derivatives of their fields, checked by central differences.
    ///
    /// Stability verdicts alone cannot see a wrong Jacobian entry that leaves the sign pattern alone,
    /// and a field that loses a factor can still come to rest at the right point. So each Jacobian is
    /// compared with a central difference of its own field at points AWAY from equilibrium, with the
    /// Wilson–Cowan model given refractory factors other than one, unequal time constants and both
    /// inputs non-zero — every term the paper's defaults would hide. Measured agreement: 3.7 × 10⁻¹¹
    /// of the largest entry for Wilson–Cowan, 2.3 × 10⁻¹⁰ at worst for `FitzHugh`'s cubic.
    #[test]
    fn the_linearisations_are_the_derivatives_of_the_fields() {
        let fd = |f: &dyn Fn(f64, f64) -> (f64, f64), x: f64, y: f64| {
            let h = 1e-6;
            let (a, b) = (f(x + h, y), f(x - h, y));
            let (c, d) = (f(x, y + h), f(x, y - h));
            [[(a.0 - b.0) / (2.0 * h), (c.0 - d.0) / (2.0 * h)], [(a.1 - b.1) / (2.0 * h), (c.1 - d.1) / (2.0 * h)]]
        };
        let close = |j: [[f64; 2]; 2], d: [[f64; 2]; 2]| {
            let scale = j.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
            j.iter().flatten().zip(d.iter().flatten()).all(|(a, b)| (a - b).abs() <= 1e-8 * scale)
        };
        for m in [Bvp::FITZHUGH_1961, Bvp { a: 0.5, b: 1.4, c: 0.7 }] {
            for (x, y, z) in [(0.3, -0.4, 0.0), (-1.7, 0.9, -0.8), (2.2, 1.5, 0.4)] {
                let j = m.linearisation(x).jacobian;
                assert!(close(j, fd(&|x, y| m.field(x, y, z), x, y)), "{m:?} at ({x}, {y}): {j:?}");
            }
        }
        let wc = WilsonCowan { re: 0.6, ri: 0.8, tau_e: 7e-3, tau_i: 13e-3, ..WilsonCowan::FIG4 };
        for (e, i, p, q) in [(0.1, 0.05, 0.3, -0.2), (0.3, 0.2, 1.0, 0.5), (-0.02, 0.1, 0.0, 0.7), (0.45, 0.4, -0.5, 0.2)] {
            let j = wc.linearisation(e, i, p, q).jacobian;
            assert!(close(j, fd(&|e, i| wc.field(e, i, p, q), e, i)), "at ({e}, {i}, {p}, {q}): {j:?}");
        }
    }

    /// Each time constant divides its own equation and only its own.
    ///
    /// Doubling `τ_i` halves `dI/dt` exactly — scaling a divisor by two commutes with rounding — and
    /// leaves `dE/dt` bit for bit; the same the other way round.
    #[test]
    fn each_time_constant_divides_only_its_own_equation() {
        let m = WilsonCowan::FIG4;
        let at = |w: WilsonCowan| w.field(0.2, 0.1, 0.3, 0.4);
        let (de, di) = at(m);
        assert!(de != 0.0 && di != 0.0);
        assert_eq!(at(WilsonCowan { tau_i: 2.0 * m.tau_i, ..m }), (de, di / 2.0));
        assert_eq!(at(WilsonCowan { tau_e: 2.0 * m.tau_e, ..m }), (de / 2.0, di));
    }

    /// With both inputs switched on, every steady state found is a zero of the field and lies on both
    /// isoclines.
    ///
    /// The paper's figures are drawn at `Q = 0`, where a model that dropped `Q` from eq. (12) or added
    /// it with the wrong sign to eq. (14) would pass every other test. At `P = 0.3, Q = 0.5` Fig. 4's
    /// population keeps three steady states, stable–unstable–stable, with the lower one lifted off the
    /// origin; at `P = 0.5, Q = −0.3` only the upper one survives.
    #[test]
    fn with_both_inputs_on_every_steady_state_is_a_zero_of_the_field() {
        let m = WilsonCowan::FIG4;
        for (p, q, want) in [(0.3, 0.5, vec![true, false, true]), (0.5, -0.3, vec![true])] {
            let ss = m.steady_states(p, q, 20_000).unwrap();
            assert_eq!(ss.iter().map(|s| s.stable).collect::<Vec<_>>(), want, "P = {p}, Q = {q}: {ss:?}");
            for s in &ss {
                let (de, di) = m.field(s.e, s.i, p, q);
                assert!(de.abs() < 1e-9 && di.abs() < 1e-9, "{s:?}: {de} {di}");
                assert!((m.i_isocline(s.i, q).unwrap() - s.e).abs() < 1e-9, "{s:?}");
            }
            assert!(ss[0].e.abs() > 1e-3, "the inputs move every state off the origin: {:?}", ss[0]);
        }
    }

    /// The isocline's domain is exactly where it exists: just inside both ends it does, just outside
    /// it does not — for each figure and for refractory factors across the permitted range.
    #[test]
    fn the_isocline_domain_is_exactly_where_it_exists() {
        let models = [
            WilsonCowan::FIG4,
            WilsonCowan::FIG7,
            WilsonCowan::FIG11,
            WilsonCowan { re: 0.0, ..WilsonCowan::FIG4 },
            WilsonCowan { re: -0.9, ..WilsonCowan::FIG4 },
            WilsonCowan { re: 20.0, ..WilsonCowan::FIG4 },
        ];
        for m in models {
            let (lo, hi) = m.e_isocline_domain().unwrap();
            let d = 1e-9 * (hi - lo);
            assert!(m.e_isocline(lo + d, 0.0).is_some() && m.e_isocline(hi - d, 0.0).is_some(), "{m:?}");
            assert!(m.e_isocline(lo - d, 0.0).is_none() && m.e_isocline(hi + d, 0.0).is_none(), "{m:?}");
        }
        let (lo, hi) = WilsonCowan::FIG4.e_isocline_domain().unwrap();
        assert!((lo - (-0.033_569_223)).abs() < 1e-9 && (hi - 0.474_966_349).abs() < 1e-9, "({lo}, {hi})");
        // At the boundary itself the isocline is already unbounded. With `θ_e = 0` the sigmoid's
        // offset and ceiling are both exactly one half, so `r_e = 2` and `r_e = −2` put a denominator
        // at exactly zero.
        let half = WilsonCowan { se: Sigmoid { a: 1.2, theta: 0.0 }, ..WilsonCowan::FIG4 };
        assert_eq!((half.se.offset(), half.se.ceiling()), (0.5, 0.5));
        for re in [2.0, -2.0] {
            assert_eq!(
                WilsonCowan { re, ..half }.e_isocline_domain(),
                Err(PlanarError::UnboundedIsocline { re, low: -2.0, high: 2.0 })
            );
        }
        assert!(WilsonCowan { re: 1.999, ..half }.e_isocline_domain().is_ok());
    }

    /// Past the pole the isoclines still exist, and outside `(−1/k_e, 1/s₀)` the scan refuses.
    ///
    /// With `r_e = 40`, beyond Fig. 4's `1/s₀ = 29.79`, the `dE/dt = 0` curve has a second branch past
    /// the pole `E = k_e/r_e = 0.024`; at `E = 0.2` it passes through a point where `dE/dt` is zero to
    /// rounding. With `r_e = −2`, below `−1/k_e = −1.035`, it runs to `E = +∞`. In both the scan
    /// would cover a fraction of the curve, so it refuses; and the `dI/dt = 0` curve behaves the same
    /// way past ITS pole, at `r_i = 80`.
    #[test]
    fn past_the_pole_the_isoclines_still_exist_and_the_scan_refuses() {
        let far = WilsonCowan { re: 40.0, ..WilsonCowan::FIG4 };
        let i = far.e_isocline(0.2, 0.3).unwrap();
        assert!(far.field(0.2, i, 0.3, 0.0).0.abs() < 1e-9, "{i}");
        assert_eq!(far.e_isocline(0.03, 0.3), None, "between the pole and the far branch there is none");
        assert_eq!(
            far.steady_states(0.3, 0.0, 1000).unwrap_err().to_string(),
            "r_e = 40 puts the dE/dt = 0 isocline out to infinity; a finite scan needs -1.0347352589447385 < r_e < 29.78919087924268"
        );
        let neg = WilsonCowan { re: -2.0, ..WilsonCowan::FIG4 };
        let i = neg.e_isocline(5.0, 0.3).unwrap();
        assert!(neg.field(5.0, i, 0.3, 0.0).0.abs() < 1e-9, "{i}");
        assert!(matches!(neg.e_isocline_domain(), Err(PlanarError::UnboundedIsocline { re, .. }) if re == -2.0));
        let inhib = WilsonCowan { ri: 80.0, ..WilsonCowan::FIG4 };
        let e = inhib.i_isocline(0.2, 0.1).unwrap();
        assert!(inhib.field(e, 0.2, 0.0, 0.1).1.abs() < 1e-9, "{e}");
    }

    /// Every parameter is checked before a scan or a step, and the refusal names it.
    #[test]
    fn every_parameter_is_checked_and_named() {
        let base = WilsonCowan::FIG4;
        let mut cases: Vec<(WilsonCowan, &str)> = vec![
            (WilsonCowan { c1: 0.0, ..base }, "c1 = 0 must be finite and positive"),
            (WilsonCowan { c2: 0.0, ..base }, "c2 = 0 must be finite and positive"),
            (WilsonCowan { c3: 0.0, ..base }, "c3 = 0 must be finite and positive"),
            (WilsonCowan { c4: -1.0, ..base }, "c4 = -1 must be finite and positive"),
            (WilsonCowan { tau_e: 0.0, ..base }, "tau_e = 0 must be finite and positive"),
            (WilsonCowan { tau_i: f64::INFINITY, ..base }, "tau_i = inf must be finite and positive"),
            (WilsonCowan { re: f64::NAN, ..base }, "r_e = NaN is not finite"),
            (WilsonCowan { ri: f64::INFINITY, ..base }, "r_i = inf is not finite"),
        ];
        let mut se = base;
        se.se.a = 0.0;
        cases.push((se, "a_e = 0 must be finite and positive"));
        let mut si = base;
        si.si.a = f64::NAN;
        cases.push((si, "a_i = NaN must be finite and positive"));
        let mut te = base;
        te.se.theta = f64::NEG_INFINITY;
        cases.push((te, "theta_e = -inf is not finite"));
        let mut ti = base;
        ti.si.theta = f64::NAN;
        cases.push((ti, "theta_i = NaN is not finite"));
        for (m, want) in cases {
            assert_eq!(m.steady_states(0.0, 0.0, 100).unwrap_err().to_string(), want);
            assert_eq!(m.step(0.1, 0.1, 0.0, 0.0, 1e-4).unwrap_err().to_string(), want);
        }
        assert!(base.check().is_ok());
        // Thresholds and refractory factors may be zero or negative: only finiteness is asked of them.
        let mut free = WilsonCowan { re: 0.0, ri: -0.5, ..base };
        free.se.theta = -1.0;
        free.si.theta = 0.0;
        assert_eq!(free.check(), Ok(()));
    }

    /// The BVP step refuses what it cannot integrate, naming it: a parameter that is not finite, a
    /// `c` that is not positive, an infinite stimulus and an infinite step.
    #[test]
    fn the_bvp_step_refuses_what_it_cannot_integrate() {
        let m = Bvp::FITZHUGH_1961;
        let refusals = [
            (Bvp { a: f64::NAN, ..m }.step(0.0, 0.0, 0.0, 0.1), "a = NaN is not finite"),
            (Bvp { b: f64::INFINITY, ..m }.step(0.0, 0.0, 0.0, 0.1), "b = inf is not finite"),
            (Bvp { c: 0.0, ..m }.step(0.0, 0.0, 0.0, 0.1), "c = 0 must be finite and positive"),
            (m.step(0.0, 0.0, f64::INFINITY, 0.1), "z = inf is not finite"),
            (m.step(0.0, 0.0, 0.0, f64::INFINITY), "h = inf must be finite and positive"),
            (m.step(0.0, f64::NEG_INFINITY, 0.0, 0.1), "y = -inf is not finite"),
        ];
        for (got, want) in refusals {
            assert_eq!(got.unwrap_err().to_string(), want);
        }
        // Outside conditions (3) it is still an ODE, and it still steps.
        assert!(Bvp { a: 0.7, b: 2.0, c: 1.0 }.step(0.0, 0.0, 0.0, 0.1).is_ok());
    }

    /// `FitzHugh`'s conditions (3) accept what they should near each boundary: `c` below one, where
    /// `b < c²` is the binding condition and not `b < c⁴` or `b < c`.
    #[test]
    fn fitzhughs_conditions_accept_a_slow_variable_faster_than_one() {
        assert!(Bvp::new(0.7, 0.85, 0.95).is_ok(), "0.85 < 0.95² = 0.9025");
        assert!(Bvp::new(0.7, 0.85, 0.92).is_err(), "0.85 > 0.92² = 0.8464");
    }

    /// Conditions (7) and (8) are the matrix's verdict for ANY parameters, not only ones that satisfy
    /// (3) — the fields are public, and a model built outside (3) is where (8) can fail.
    ///
    /// With `b = 2`, `c = 1`, (7) holds everywhere (`2 − (1 − x₁²) > 0`) and (8) fails for
    /// `|x₁| < 1/√2`, so near the origin the point is unstable on (8) alone.
    #[test]
    fn fitzhughs_conditions_are_the_matrix_verdict_for_any_parameters() {
        for m in [Bvp { a: 0.7, b: 2.0, c: 1.0 }, Bvp { a: 0.5, b: 1.4, c: 0.7 }, Bvp { a: 0.2, b: 0.3, c: 0.5 }] {
            for n in -250..=250 {
                let x1 = f64::from(n) / 100.0;
                assert_eq!(m.is_stable(x1), m.linearisation(x1).is_stable(), "{m:?} at {x1}");
            }
        }
        assert!(!Bvp { a: 0.7, b: 2.0, c: 1.0 }.is_stable(0.0), "(8) fails at the origin");
    }

    /// At the sigmoid's ceiling there is no finite preimage: where `y + s₀` is exactly one, the inverse
    /// refuses rather than returning `+∞`; where it is exactly zero, rather than `−∞`.
    #[test]
    fn the_inverse_refuses_both_ends_of_the_range() {
        for s in [WilsonCowan::FIG4.se, WilsonCowan::FIG7.si, WilsonCowan::FIG11.se] {
            // The input whose `u` is exactly one lies within a few units in the last place of the
            // ceiling. The search is bounded: an unbounded one hung for fifteen minutes under a
            // mutation that set the ceiling to one, stepping down from it one ulp at a time.
            let mut y = s.ceiling();
            for _ in 0..8 {
                if y + s.offset() < 1.0 {
                    y = y.next_up();
                } else if y + s.offset() > 1.0 {
                    y = y.next_down();
                }
            }
            assert_eq!(y + s.offset(), 1.0, "{s:?}: no input within 8 ulps of the ceiling reaches u = 1");
            assert_eq!(s.inverse(y), None, "{s:?} at its ceiling");
            assert_eq!(-s.offset() + s.offset(), 0.0);
            assert_eq!(s.inverse(-s.offset()), None, "{s:?} at its floor");
        }
    }

    /// The four figures' parameters, pinned by value against the paper.
    ///
    /// Several transcription slips move no steady-state COUNT — Fig. 4 with `θ_e = 2.3` still has
    /// three, stable–unstable–stable — so the values themselves are the check.
    #[test]
    fn the_figures_parameters_are_the_papers() {
        let row = |m: WilsonCowan| [m.c1, m.c2, m.c3, m.c4, m.se.a, m.se.theta, m.si.a, m.si.theta, m.re, m.ri, m.tau_e, m.tau_i];
        assert_eq!(row(WilsonCowan::FIG4), [12.0, 4.0, 13.0, 11.0, 1.2, 2.8, 1.0, 4.0, 1.0, 1.0, 0.010, 0.010]);
        assert_eq!(row(WilsonCowan::FIG6), [13.0, 4.0, 20.0, 2.0, 1.2, 2.7, 5.0, 3.7, 1.0, 1.0, 0.010, 0.010]);
        assert_eq!(row(WilsonCowan::FIG7), [13.0, 4.0, 22.0, 2.0, 1.5, 2.5, 6.0, 4.3, 1.0, 1.0, 0.010, 0.010]);
        assert_eq!(row(WilsonCowan::FIG11), [16.0, 12.0, 15.0, 3.0, 1.3, 4.0, 2.0, 3.7, 1.0, 1.0, 0.008, 0.008]);
        assert_eq!(WilsonCowan::FIG11_P, 1.25);
        assert_eq!((Bvp::FITZHUGH_1961.a, Bvp::FITZHUGH_1961.b, Bvp::FITZHUGH_1961.c), (0.7, 0.8, 3.0));
    }

    /// Conditions (17) and (18) at their boundaries.
    ///
    /// (17) is strict: with `a_e = 1.5`, `c₁ = 6` is exactly `9/a_e` and fails, and the next double up
    /// passes. (18) needs `a_e c₁ > 9` as a precondition, and the paper says so: at `a_e c₁ = 9` its
    /// left side is `x/0 = +∞`, which would pass a bare comparison although the kink it needs is
    /// absent. And (18) compares `(a_i c₄ + 9)/(a_i c₃)` — with `c₃ = 20` it holds, `1 > 0.5`, and a
    /// right side built on `c₁` instead would give `1 > 1`.
    #[test]
    fn conditions_17_and_18_at_their_boundaries() {
        let mut m = WilsonCowan::FIG7;
        m.c1 = 6.0;
        assert!(!m.kink_condition(), "6 > 9/1.5 = 6 is false");
        assert!(!m.five_state_condition(), "a_e c1 = 9: the precondition fails");
        m.c1 = 6.0_f64.next_up();
        assert!(m.kink_condition());
        let mut five = WilsonCowan { c1: 10.0, c2: 1.0, c3: 20.0, c4: 1.0, ..WilsonCowan::FIG4 };
        five.se.a = 1.0;
        five.si.a = 1.0;
        assert!(five.five_state_condition(), "1/(10 − 9) = 1 > (1 + 9)/20");
        assert!(!WilsonCowan { c3: 5.0, ..five }.five_state_condition(), "1 > 10/5 is false");
        assert!(!WilsonCowan::FIG4.five_state_condition(), "Fig. 4 has three steady states, not five");
        assert!(WilsonCowan { c1: 9.0, ..WilsonCowan::FIG4 }.kink_condition(), "9 > 9/1.2 = 7.5");
    }
}
