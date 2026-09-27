//! Hindmarsh and Rose's bursting neuron: the two-variable model with three equilibrium points, and
//! the third, slow equation of adaptation that turns its triggered firing into bursts.
//!
//! # The paper
//!
//! Hindmarsh and Rose, *A model of neuronal bursting using three coupled first order differential
//! equations*, Proceedings of the Royal Society of London B 221:87–102, 1984
//! (doi:10.1098/rspb.1984.0024). Starting from `FitzHugh`'s BVP model, they bend the steady-state
//! current–voltage curve until it crosses zero three times, which puts three equilibrium points
//! (e.p.s) in the phase plane — a stable rest state, a saddle and an unstable spiral inside a stable
//! limit cycle — and then add a slow outward current that switches the model between the two:
//!
//! ```text
//! ẋ = y − a x³ + b x² + I            (13)   [(7) when I = 0]
//! ẏ = c − d x² − y                   (14)   [(8)]
//!
//! ẋ = y − a x³ + b x² + I − z        ⎫
//! ẏ = c − d x² − y                   ⎬ (15)
//! ż = r (s (x − x₁) − z)             ⎭
//! ```
//!
//! `x` is the membrane potential, `y` the recovery variable and `z` the adaptation current, all
//! dimensionless, as is time ("time units"). `(x₁, y₁)` is "the leftmost e.p. of the model without
//! adaptation" (p. 96), so that `z` rests at zero. The paper's one numerical set, used for every
//! figure, is `a = 1, b = 3, c = 1, d = 5` ([`TwoVariable::PAPER`]; Fig. 3 caption p. 93, Fig. 4
//! caption p. 94, p. 96, p. 98); the adaptation constants are `r = 0.001` with `s = 1`
//! ([`ThreeVariable::FIG5A`]) or `s = 4` (Figs 5c, 6 and 8), and `r = 0.005, s = 4` for the random
//! bursts of p. 98 ([`ThreeVariable::RANDOM_BURSTS`]). The currents are a pulse of `I = 1` (Figs 3a,
//! 5a and 5c), steady `I = 0.4`, 2 and 4 (Fig. 6), a step of `I = −3` (Fig. 8) and `I = 3.25`
//! (p. 98). No pulse or step duration is printed as a number; p. 99 gives Fig. 8's step only as
//! "a period similar to the burst duration".
//!
//! # What is exact, and where the current enters it
//!
//! Setting `ẏ = 0` in (14) puts an e.p. on the parabola `y = c − d x²`, and `ẋ = 0` in (13) then reads
//! `c − d x² − a x³ + b x² + I = 0`, that is `a x³ + (d − b) x² = c + I`:
//!
//! ```text
//! x³ + p x² = q,    p = (d − b)/a,    q = (c + I)/a                       (9)
//! ```
//!
//! The paper writes (9) before it adds the current, with `q = c/a`; the current only raises the
//! horizontal line of height `q` in its Fig. 2, and `p` does not move. The module finds the roots
//! in closed form ([`TwoVariable::equilibria`]); for the paper's constants at `I = 0` the cubic
//! factors as `(x + 1)(x² + x − 1)`, so the three e.p.s are EXACTLY
//!
//! ```text
//! x₁ = −(1 + √5)/2 = −1.618 033 988 749 895,    −1,    (√5 − 1)/2,    y₁ = c − d x₁² = −(13 + 5√5)/2
//! ```
//!
//! — the golden ratio, which the paper never prints. The same algebra gives the steady-state
//! current–voltage curve the paper draws above Fig. 4 without writing it down,
//! `I(∞) = a x³ + (d − b) x² − c` ([`TwoVariable::steady_current`]): the current that holds `x` at
//! rest, whose zeros are the e.p.s at `I = 0` (p. 95).
//!
//! What pp. 91–92 derive, and this module implements and checks:
//!
//! - for `q > 0`, which is the paper's `c > 0` at `I = 0`, three e.p.s exactly when `27q < 4p³`
//!   ([`TwoVariable::three_point_condition`], which tests `0 < 27q < 4p³` as p. 92 writes it),
//!   "which in turn requires that `b < d`" (p. 91);
//! - the linearisation `A(x₀) = [[−3a x₀² + 2b x₀, 1], [−2d x₀, −1]]` with
//!   `Tr = −3a x₀² + 2b x₀ − 1` and `Det = 3a x₀² + 2(d − b) x₀` ([`TwoVariable::trace`],
//!   [`TwoVariable::determinant`]) — and the determinant is the SLOPE of `I(∞)`, so the saddle is
//!   exactly the e.p. on the negative-slope limb of Fig. 4's curve;
//! - Table 1's five regions of `x₀` with `D = (b² − 3a)^½`, their printed signs of `Tr` and `Det` and
//!   their printed e.p. types ([`TwoVariable::region`], [`Region`]);
//! - the band of region IV, `L = (b − D)/3a` (10) and `M = (b + D)/3a` (11), and condition (12),
//!   `L³ + pL² < q < M³ + pM²`, for the rightmost e.p. to lie in it
//!   ([`TwoVariable::region_iv_condition`]), shown equivalent to `L < C < M` for the root itself.
//!
//! For the paper's constants that puts `A = x₁` in region I (a stable node), `B = −1` in region II
//! (a saddle) and `C = (√5 − 1)/2` in region IV (an unstable spiral), which is what Fig. 4 labels
//! them. Because `(x₁, y₁)` is an e.p. of (13)–(14), `(x₁, y₁, 0)` is an e.p. of (15) — exactly, at
//! `I = 0` — and the tests show it is stable for all three figure sets by the Routh–Hurwitz
//! conditions on [`ThreeVariable::characteristic`], as p. 96 asserts.
//!
//! ⚠ **The paper never prints `x₁`, and the `−1.6` it prints is a rounding.** It locates
//! the e.p.s "at x values of −1.6, −1 and +0.6" (p. 94) and starts Fig. 3a "in equilibrium at
//! x = −1.6" (p. 95); `x₁` itself is defined only as "the leftmost e.p. of the model without
//! adaptation" (p. 96), and that definition settles it: the leftmost root is [`ThreeVariable::X1`].
//! The rounded [`ThreeVariable::X1_PRINTED`] is not an equilibrium of anything here: at
//! `(−1.6, −11.8, 0)` the `x` equation of (15) reads `ẋ = −0.024` at `I = 0`, and with `x₁ = −1.6` in
//! the `z` equation the true rest moves — to `x = −1.6045` with `z = −0.018` for `s = 4` (Figs 5c, 6
//! and 8), and to `x = −1.6104` with `z = −0.0104` for Fig. 5a's `s = 1` — so the "resting"
//! adaptation current is not zero. The rounding moves the dynamics measurably: the Fig. 6b burst
//! period falls from 452.84 to 430.78 time units (−4.9%), the first long burst gains a spike (69 to
//! 70), Fig. 6c's steady interspike interval falls from 22.066 to 20.735, and Fig. 6a, eight spikes
//! either way, has the ratio of its last to its first interspike interval fall from 2.504 to 2.244.
//!
//! The drawings do not settle which value the authors ran: read against runs of both, they point
//! different ways (the rounding run, here and below, from `(−1.6, −11.8, 0)` as p. 95 starts
//! Fig. 3a, unless its own rest is named). Fig. 6a's last interspike interval over its first, which
//! needs no time scale, is 41.0 px over 15.0 or 15.5 at the `x = 1` row of the 600 ppi scan of
//! p. 98, 2.73 or 2.65 — 2.50 to 3.0 at one pixel either way, and 2.41 to 2.83 as the row moves
//! between `x ≈ 0.65` and 1.4 — which takes in the exact root's 2.504 and not the rounding's 2.244.
//! Fig. 5c draws one spike after a pulse 11.0 or 11.6 wide by the two readings of its bar on the
//! p. 96 scan: so does the exact root, and so does the rounding, but the rounding started at its
//! own rest fires twice from a pulse of 10.92, below both, and at every quarter from 11 to 13.5. A
//! periodic burst's duration over the burst period is, for the two bursts of Fig. 6b,
//! 0.3084–0.3110 and 0.3066–0.3091 over the rows from `x ≈ 0.65` to 1.4 of that panel, against
//! 0.3113 and 0.3025: between the two, the first burst's nearer the exact root and the second's
//! around their midpoint, 0.3069. The first drawn spike of Fig. 6b, which the caption starts "700
//! time units after the onset of I step", comes 209 px of a 642 px period into the panel, 0.3255 of
//! it, against 0.3369 and 0.3190: nearer the rounding. So is Fig. 8's count: its nine rebound spikes,
//! after a step drawn 159–169 long, need a step of 184 with the exact root, 176 with the rounding
//! and 170 with the rounding started at its rest (below). Fig. 6c does not discriminate: its
//! interval, 0.0440 of Fig. 6b's period, is one both runs pass through on the way to their steady
//! intervals (below). This module uses the exact root, on the strength of p. 96's definition, and
//! keeps the printed value as a named constant, so a reader can run both.
//!
//! ⚠ **Fig. 6a's after-hyperpolarization is not what the printed equations give.** The text says
//! the burst is "followed by an after-hyperpolarizing wave which slowly recovered to the starting
//! value (x₁)" (p. 97), and the drawing agrees with it. Read on the p. 98 scan against the ticks at
//! 2, 0 and −2, and timed by the "100" bar between its tick centres (142.5 px) from the last spike,
//! the drawn tail stays between −1.68 and −1.65 from t ≈ 220 to t ≈ 1 350 and ends at −1.63 at
//! t ≈ 1 430: a slow recovery towards `x₁`, from below. Under the caption's steady `I = 0.4` the one
//! e.p. of (15) is at `x = −1.5406`, which is `x₁ + 0.0774`, and the model goes there: it dips to
//! −1.7006 at t = 224.32 and is back at −1.672 by t = 300, −1.634 by 400, −1.553 by 856 and −1.542
//! by 1 205. Between t = 300 and 856 the model rises 0.12 where the drawing rises 0.02, and it
//! settles 0.077 ABOVE `x₁`. The onset differs the same way: the drawn trace starts at `x ≈ −1.46`
//! 87–89.5 px before its first spike, 57–63 time units by the two readings of the bar, where the
//! model takes 34.8 from `x = −1.46` to its first spike. The text's "recovered to the starting
//! value" is exact for the pulse of Fig. 5a, where the current returns to zero; the drawn tail of
//! Fig. 6a is a figure the printed equations do not reproduce.
//!
//! ⚠ **The paper's `q > 0` is lost once there is a current.** p. 91 argues "`q > 0` since `a > 0`
//! and `c > 0`", before (13) adds `I`; with `q = (c + I)/a` any `I < −c` makes `q` negative — Fig. 8's
//! `I = −3` gives `q = −2`. The count condition in full is the discriminant of (9),
//! `q(4p³ − 27q) > 0` ([`TwoVariable::discriminant`]), and `0 < 27q < 4p³` is its `q > 0` half; the
//! other half, `4p³ < 27q < 0`, gives three e.p.s with `b > d`, which the paper's "requires that
//! `b < d`" excludes. Likewise "one e.p. if `p < 0`" (p. 91) is sufficient only while `q > 0` — that
//! same half gives three e.p.s with `p < 0`, as `a = 1, b = 5, c = 1, d = 2` does under `I = −3` —
//! and it is not necessary: `p > 0` with `27q > 4p³` has one too, which is the case for every
//! current of Fig. 6. And condition (12) places "the positive root", which for `q > 0` exists and
//! is unique whatever `p` is. At `q ≤ 0` there is none while `b < d`, but with `b > d` there can be:
//! the same `a = 1, b = 5, c = 1, d = 2` has one at `q = 0`, under `I = −1`, where (9) is
//! `x²(x − 3)`; two under `I = −3`; and a double one at `27q = 4p³`, under `I = −5`, where (9) is
//! `(x + 1)(x − 2)²`. Table 1 and §(c) presuppose `b < d`, so [`TwoVariable::region_iv_condition`]
//! refuses `b ≥ d` as [`TwoVariable::region`] does, and then a current that makes `q ≤ 0`, which
//! leaves no positive root at all.
//!
//! ⚠ **Table 1 gives one sign pattern two names in three rows, and the sign pattern cannot decide
//! the name.** Regions I, III and V all have `Tr < 0`, `Det > 0`; the table calls I a "stable node
//! or spiral" and III and V a "stable focus or spiral", and region IV is an "unstable focus or
//! spiral" there and on p. 91 but "an unstable node or spiral" on p. 92. A node and a focus differ
//! in the sign of
//! `Tr² − 4 Det`, which the table does not use: the module keeps the printed words
//! ([`Region::printed_type`]) and classifies each e.p. separately by its eigenvalues
//! ([`TwoVariable::kind`]). The table also presupposes `b < d` — otherwise the saddle interval
//! `−2(d − b)/3a < x₀ < 0` is empty and the negative determinant moves to the positive side, over
//! region III — so [`TwoVariable::region`] refuses a model with `b ≥ d`, and one with `b² ≤ 3a`,
//! where `D` is not real or is zero and region IV is empty.
//!
//! ⚠ **Figs 3a, 5a and 8, the tail of Fig. 6a (above), and the random-burst sequence cannot be
//! reproduced from the printed equations and parameters**, by measurements of the 600 ppi scans and
//! by the runs in this module's tests, where `SciPy`'s `DOP853` gives the same counts as the
//! module's RK4.
//!
//! - Fig. 5a shows an eight-spike triggered burst, every spike after a pulse drawn 11–12 time units
//!   wide (the width is not printed). Pulses from 8 to 24 give at most five spikes, and the drawn
//!   width gives four. On a grid of quarters the first pulse to give eight is 44.75, about four
//!   times the drawn one, and five of its eight fall inside the pulse, which the drawing does not
//!   show.
//! - Fig. 8 shows nine rebound spikes after a step of `I = −3` held "for a period similar to the
//!   burst duration" (p. 99), the first of them 38.2–40.7 time units after the step ends by the two
//!   readings of its "20" bar, and the step drawn 159–169 long. A periodic burst of Fig. 6b lasts
//!   140.98 time units, and a step of 141 gives six. Over the drawn range the exact root gives seven
//!   (steps to 164) or eight, and nine from a step of 184; the rounding nearly gives the count, nine
//!   from a step of 176, or of 170 started at its own rest, where 169 gives eight. The latency is
//!   what no run reproduces. On steps of whole time units from 140 to 226 it falls at every step,
//!   and nine spikes come no sooner than 43.51 after release with the exact root (a step of 204),
//!   44.27 with the rounding (194) and 44.28 with the rounding from its rest (188); the first step
//!   whose spike comes within 40.7 — 220, 213 and 207 — gives ten. No step of the three sweeps gives
//!   the drawn count and the drawn latency together.
//! - Fig. 3a's spikes are 25.6–28.5 units apart by its own "30" bar, but the limit cycle of
//!   (13)–(14) has a period of 18.63. Whatever the bar, the drawn pulse (53 px) is about a third
//!   of a drawn interspike interval (156–167 px), where the model needs a pulse of at least 9.89
//!   time units, 0.53 of its period, to switch onto the limit cycle at all. And the resting trace is
//!   drawn at `x ≈ −1.48` by the 0 and −2 ticks of the p. 93 scan, not at the −1.6 of p. 95 or the
//!   `x₁ = −1.618` of the model, while the drawn spikes peak near 1.7 by the 2 and 0 ticks, as the
//!   limit cycle does at 1.686.
//! - The random-burst sequence of p. 98 — "7, 5, 5, 7, 5, 5, 5, 5, 6, 4, 3, 3, 7, 5, 4, 5, 7" —
//!   depends on an initial state, an integrator and a rule for where one burst ends, none of which is
//!   printed, in a regime that amplifies differences: a change of `10⁻¹⁵` in `z` moves a spike by a
//!   whole time unit within 2 600 time units, and RK4 and `DOP853` run from the same state agree on
//!   the first eight bursts and not after. The tests show the burst sizes are irregular — 3 to 7 in
//!   the module's run, most often 5 and then 7, as in the printed seventeen, though `DOP853`'s run
//!   includes a burst of two — and do not claim the sequence.
//!
//! ⚠ **Fig. 6c is drawn later than its caption says.** The caption starts the panel as it starts
//! panel b, "(b) starts 700 time units after the onset of I step, and (c) after 1000 time units of
//! continuous firing" (p. 98), and p. 97 calls it "the steady repetitive firing". Its twelve spikes
//! average 28.2 px apart on the p. 98 scan: 0.0440 of Fig. 6b's 642 px period, a ratio that needs
//! no time scale, or 19.8 and 18.7 time units by the two readings of the "100" bar
//! (142.5 px between its tick centres, 151 px end to end). The model's twelve spikes from t = 1 000
//! average 0.0255 of the period, 11.53 time units, and its steady interval, 22.066, is 0.0487 of
//! it: neither is drawn. What is drawn is the model's firing some 500 time units later. The first
//! twelve of its spikes to average 0.0440 of the period run from t = 1 491.7 to 1 712.6, and the
//! first single interval that long starts at t = 1 569.05; by the bar's two readings, the first
//! interval of 18.7 starts at t = 1 491.7 and the first of 19.8 at 1 569.05. With the rounding the
//! twelve run from 1 550.9 to 1 760.7 and the interval starts at 1 624.96. The drawing is
//! reproducible; the caption's "1000" is not, and the firing drawn is not yet the "steady" firing
//! of p. 97, the interval still growing there towards 22.066 (20.735 with the rounding).
//!
//! Smaller slips, recorded so nobody "corrects" the model to them: eq. (1) is printed with `βr` where
//! the transformation to (5)–(6) needs `βw`, `w = r + z` with `z` the slow inward current of eq. (3)
//! (p. 89), not the adaptation current that (15) calls `z`; the scaling on p. 90 mixes `T` and `t`
//! (`x(T) = v(T)`, `y(T) = (αβ/γδ)w(t)`); p. 92 says "the conditions (10) and (11)" of what are
//! definitions — the condition is (12); and the Fig. 7 caption's "(d) to (e)" covers three panels,
//! (d) to (f), as the text's reference to "figure 7f" shows.
//!
//! # What is checked
//!
//! The closed forms against each other and against the field: every root of (9) is a zero of
//! (13)–(14) for currents on both sides of each count boundary and on both boundaries, the count
//! against the sign of the discriminant at every point of a sweep that reaches `|q| = 3·2⁻⁶⁴`, the
//! double root at the origin exactly at `q = 0` for 321 values of `p`, the printed `Tr` and `Det`
//! against the matrix and the matrix against central differences, Table 1 region by region with
//! `L`, `M` and `−2(d − b)/3a` the zeros they bound, (12) against the root it locates and its
//! refusal where there is none, the golden-ratio roots to the last bit and proved the nearest
//! doubles, every branch of the cubic solver at a point where it is exact or where its limit shows,
//! and the Jacobian and characteristic polynomial of (15). The integrator is fourth order,
//! measured. The figures the equations reproduce: Fig. 4's three e.p. types, Fig. 3a's triggered
//! switch onto the limit cycle (not its timing), band by band over every pulse to 40 in hundredths,
//! Fig. 5c's single spike after the pulse it draws, with the dip below `x₁` after it, Fig. 6a's
//! isolated eight-spike burst at `I = 0.4` (not its tail), Fig. 6b's 69-spike burst followed by
//! nine-spike bursts every 452.84 time units at `I = 2`, and Fig. 6c's drawn interval (not its
//! caption's time), with the exact `x₁` and the printed one. The runs behind the figures they do
//! not reproduce, each with the measurement that says so: Fig. 5a's short bursts, Fig. 6a's tail,
//! Fig. 8's rebound latency, and the irregular bursts of p. 98.
//!
//! There is no authors' implementation to run: the paper thanks D. A. Evans "for assistance with
//! the numerical work" (p. 101) and prints no program. Every figure's spike times, spike counts,
//! dips and switching threshold are checked against `SciPy`'s eighth-order `DOP853` at a tolerance
//! of `10⁻¹²` on the same inputs, and every equilibrium and eigenvalue against `NumPy`'s
//! companion-matrix roots and `eigvals` — all from `tools/hindmarshrose_reference.py`. `DOP853` is
//! itself checked against the implicit Radau method on the six Fig. 6 runs (both roots, three
//! currents), where the two agree to within 7 × 10⁻⁹. On the random-burst run they part company as
//! RK4 and `DOP853` do — Radau counts 209 spikes in 6 000 time units where `DOP853` counts 208 — so
//! past its first eight bursts that run is the module's own.
//!
//! No integrator, step or spike rule is printed anywhere in the paper. [`ThreeVariable::simulate`]
//! uses classical Runge–Kutta at a step the caller chooses and counts a spike at each upward crossing
//! of a threshold the caller chooses, interpolated linearly inside the step; the tests use
//! `h = 0.01` and `x = 1`, and start every run with the exact root at the rest state
//! `(x₁, y₁, 0)`, and every run with the rounding at `(−1.6, −11.8, 0)` unless it names that
//! model's own rest.

use core::fmt;

use crate::planar::Linearisation;

/// Why a Hindmarsh–Rose question could not be answered.
#[derive(Debug, Clone, PartialEq)]
pub enum HindmarshRoseError {
    /// A parameter that must be finite and positive was not.
    NotPositive {
        /// Which parameter.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A state, current, abscissa, threshold or spike time that is not a finite number.
    NonFinite {
        /// Which quantity.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// `b² ≤ 3a`: `D = (b² − 3a)^½` is not real or is zero, region IV is empty, and Table 1 does not
    /// apply.
    NoUnstableBand {
        /// `a`.
        a: f64,
        /// `b`.
        b: f64,
    },
    /// `b ≥ d`: Table 1's saddle interval is not left of the origin, so its regions overlap.
    RegionsOutOfOrder {
        /// `b`.
        b: f64,
        /// `d`.
        d: f64,
    },
    /// `q = (c + I)/a ≤ 0` with `b < d`: `x³ + px²` is then positive for every `x > 0`, so (9) has
    /// no positive root for condition (12) to place.
    NoPositiveRoot {
        /// `q`.
        q: f64,
    },
    /// The integration left the finite numbers: the step is too long for the fast equations.
    Diverged {
        /// The time at the start of the step that failed.
        t: f64,
        /// The step.
        h: f64,
    },
    /// A spike train whose times decrease, which no run produces and no burst rule can split.
    Unordered {
        /// The index of the spike that comes before its predecessor.
        index: usize,
        /// The predecessor's time.
        before: f64,
        /// This spike's time.
        after: f64,
    },
}

impl fmt::Display for HindmarshRoseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositive { what, value } => write!(f, "{what} = {value} must be finite and positive"),
            Self::NonFinite { what, value } => write!(f, "{what} = {value} is not finite"),
            Self::NoUnstableBand { a, b } => {
                write!(f, "a = {a}, b = {b} give b^2 <= 3a, and Table 1 and eqs. (10)-(11) need b^2 > 3a")
            }
            Self::RegionsOutOfOrder { b, d } => {
                write!(f, "b = {b} is not below d = {d}, and Table 1's regions are in order only when b < d")
            }
            Self::NoPositiveRoot { q } => {
                write!(f, "q = {q} is not positive, so with b < d eq. (9) has no positive root for condition (12) to place")
            }
            Self::Diverged { t, h } => write!(f, "the state left the finite numbers at t = {t}: the step h = {h} is too long"),
            Self::Unordered { index, before, after } => {
                write!(f, "spike {index} at t = {after} comes before the spike ahead of it at t = {before}")
            }
        }
    }
}

impl std::error::Error for HindmarshRoseError {}

fn finite(what: &'static str, value: f64) -> Result<f64, HindmarshRoseError> {
    if value.is_finite() { Ok(value) } else { Err(HindmarshRoseError::NonFinite { what, value }) }
}

fn positive(what: &'static str, value: f64) -> Result<f64, HindmarshRoseError> {
    if value.is_finite() && value > 0.0 { Ok(value) } else { Err(HindmarshRoseError::NotPositive { what, value }) }
}

/// The real roots of `x³ + b x² + c x + d`, ascending, each distinct root once where the formulas
/// can tell them apart.
///
/// A zero constant term is factored out exactly: `x(x² + b x + c)` gives the origin, once, and the
/// quadratic's roots. Otherwise the case is decided by the cubic's own discriminant, factored as
/// `Δ = d(18bc − 4b³ − 27d) + c²(b² − 4c)`. For (9) — `b = p`, `c = 0`, `d = −q` — that is
/// `q(4p³ − 27q)` computed operation for operation as [`TwoVariable::discriminant`] computes it, so
/// while `p³` is finite the number of e.p.s follows that sign exactly. The roots come from the
/// depressed cubic `t³ + P t + Q`, `x = t − b/3`: three by the trigonometric formula when `Δ > 0`
/// and `P < 0`, one by Cardano's with its radicand `−Δ/108` when `Δ < 0`, and otherwise a double
/// root beside a simple one, or a triple root where `P` is not negative. Each root then takes one
/// Newton step on the undepressed cubic, which lands the three roots of the paper's cubic bit for
/// bit on `−(1 + √5)/2`, `−1` and `(√5 − 1)/2` evaluated in `f64`.
///
/// ⚠ The depressed discriminant `−(4P³ + 27Q²)` is the same number in exact arithmetic, and it is
/// what this solver used to decide by. For (9) it is two terms of size `4p⁶/27` cancelling down to
/// `q(4p³ − 27q)`, and at `q = 0` — the current `I = −c`, the edge of the paper's `0 < 27q` — it
/// rounds to `−7.3 × 10⁻¹²` for `p = 7` and `+1.1 × 10⁻¹⁶` for `p = 5/4` instead of to zero, so the
/// solver dropped the e.p. at the origin, or split it into two that are not there.
///
/// ⚠ Roots closer than about `√ε` of the cubic's scale are not resolved, because there `Δ` itself
/// is a difference of rounded terms. Near `q = 0` the count is still right — it is the paper's
/// discriminant — but the two small roots `±√(q/p)` come out together: for `p = 7/8, q = 2⁻⁶⁰` the
/// trigonometric formula's argument rounds past −1 (which is why it is clamped) and both land on
/// `0`. With a linear term the count can go too: for roots `−1`, `−1 + 2⁻²⁶` and `15/8`, whose
/// coefficients are stored exactly, `Δ` is lost in the rounding of its terms and the pair comes
/// back once, at its midpoint. And near a triple root `Δ` and `P` can disagree — `Δ > 0` with
/// `P ≥ 0`, which three real roots never allow — and the three come back as one.
///
/// ⚠ The Newton step is kept only when it does not raise the residual. At such an unresolved pair
/// the root sits near the cubic's turning point between the two, where the slope is nearly zero,
/// and an unguarded step throws it far away.
fn cubic_roots(b: f64, c: f64, d: f64) -> Vec<f64> {
    let mut roots: Vec<f64> = if d == 0.0 {
        let mut roots = vec![0.0];
        roots.extend(quadratic_roots(b, c).into_iter().filter(|&x| x != 0.0));
        roots
    } else {
        let shift = b / 3.0;
        let p = c - b * b / 3.0;
        let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
        let disc = d * (18.0 * b * c - 4.0 * b * b * b - 27.0 * d) + c * c * (b * b - 4.0 * c);
        let depressed: Vec<f64> = if disc > 0.0 && p < 0.0 {
            let m = 2.0 * (-p / 3.0).sqrt();
            let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
            (0..3).map(|k| m * (theta - 2.0 * core::f64::consts::PI * f64::from(k) / 3.0).cos()).collect()
        } else if disc < 0.0 {
            let root = (-disc / 108.0).sqrt();
            vec![(-q / 2.0 + root).cbrt() + (-q / 2.0 - root).cbrt()]
        } else if p < 0.0 {
            vec![3.0 * q / p, -3.0 * q / (2.0 * p)]
        } else {
            vec![0.0]
        };
        depressed
            .into_iter()
            .map(|t| {
                let x = t - shift;
                let residual = |x: f64| ((x + b) * x + c) * x + d;
                let polished = x - residual(x) / ((3.0 * x + 2.0 * b) * x + c);
                if polished.is_finite() && residual(polished).abs() <= residual(x).abs() { polished } else { x }
            })
            .collect()
    };
    roots.sort_by(f64::total_cmp);
    roots
}

/// The real roots of `x² + b x + c`, unordered: the larger in size from the formula whose two
/// terms have the same sign, the other as `c` over it.
fn quadratic_roots(b: f64, c: f64) -> Vec<f64> {
    let spread = b * b - 4.0 * c;
    if spread > 0.0 {
        let far = -0.5 * (b + spread.sqrt().copysign(b));
        vec![far, c / far]
    } else if spread == 0.0 {
        vec![-0.5 * b]
    } else {
        Vec::new()
    }
}

/// One classical fourth-order Runge–Kutta step of an autonomous field on `N` variables.
fn rk4<const N: usize>(f: impl Fn([f64; N]) -> [f64; N], u: [f64; N], h: f64) -> [f64; N] {
    let at = |k: [f64; N], w: f64| -> [f64; N] { core::array::from_fn(|n| u[n] + w * k[n]) };
    let k1 = f(u);
    let k2 = f(at(k1, 0.5 * h));
    let k3 = f(at(k2, 0.5 * h));
    let k4 = f(at(k3, h));
    core::array::from_fn(|n| u[n] + h / 6.0 * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n]))
}

/// What a run produced: the spike times and the state it ended in.
#[derive(Debug, Clone, PartialEq)]
pub struct Run<const N: usize> {
    /// Each upward crossing of the threshold by `x`, in time units from the start of the run,
    /// interpolated linearly inside the step that made it.
    pub spikes: Vec<f64>,
    /// The state after the last step.
    pub end: [f64; N],
}

/// Integrate `field` through a protocol of `(steps, current)` segments, counting spikes.
fn integrate<const N: usize>(
    field: impl Fn([f64; N], f64) -> [f64; N],
    start: [f64; N],
    protocol: &[(usize, f64)],
    h: f64,
    threshold: f64,
) -> Result<Run<N>, HindmarshRoseError> {
    positive("h", h)?;
    finite("threshold", threshold)?;
    for (what, value) in ["x", "y", "z"].into_iter().zip(start) {
        finite(what, value)?;
    }
    for &(_, i) in protocol {
        finite("I", i)?;
    }
    let (mut u, mut k, mut spikes) = (start, 0_usize, Vec::new());
    for &(steps, i) in protocol {
        for _ in 0..steps {
            let next = rk4(|v| field(v, i), u, h);
            if !next.iter().all(|v| v.is_finite()) {
                return Err(HindmarshRoseError::Diverged { t: k as f64 * h, h });
            }
            if u[0] < threshold && next[0] >= threshold {
                spikes.push((k as f64 + (threshold - u[0]) / (next[0] - u[0])) * h);
            }
            u = next;
            k += 1;
        }
    }
    Ok(Run { spikes, end: u })
}

/// The sizes of successive bursts in a spike train: a new burst starts wherever two consecutive
/// spikes are more than `gap` apart.
///
/// The paper prints no rule for where one burst ends, and in the random-burst regime of p. 98 the
/// sizes depend on it; the tests say which gap they use.
///
/// # Errors
///
/// [`HindmarshRoseError::NotPositive`] for a `gap` that is not finite and positive;
/// [`HindmarshRoseError::NonFinite`] for a spike time that is not finite;
/// [`HindmarshRoseError::Unordered`] for a spike earlier than the one before it.
pub fn burst_sizes(spikes: &[f64], gap: f64) -> Result<Vec<usize>, HindmarshRoseError> {
    positive("gap", gap)?;
    for (n, &t) in spikes.iter().enumerate() {
        finite("spike time", t)?;
        if n > 0 && t < spikes[n - 1] {
            return Err(HindmarshRoseError::Unordered { index: n, before: spikes[n - 1], after: t });
        }
    }
    let (mut sizes, mut first) = (Vec::new(), 0);
    for n in 1..spikes.len() {
        if spikes[n] - spikes[n - 1] > gap {
            sizes.push(n - first);
            first = n;
        }
    }
    if !spikes.is_empty() {
        sizes.push(spikes.len() - first);
    }
    Ok(sizes)
}

/// Where an e.p.'s abscissa lies among Table 1's five regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// `x₀ < −2(d − b)/3a`.
    I,
    /// `−2(d − b)/3a < x₀ < 0`.
    II,
    /// `0 < x₀ < (b − D)/3a`.
    III,
    /// `(b − D)/3a < x₀ < (b + D)/3a`: the band where the trace is positive.
    IV,
    /// `(b + D)/3a < x₀`.
    V,
}

impl Region {
    /// Table 1's sign of `Tr(A(x₀))`: positive in region IV only.
    #[must_use]
    pub fn trace_positive(self) -> bool {
        self == Self::IV
    }

    /// Table 1's sign of `Det(A(x₀))`: negative in region II only.
    #[must_use]
    pub fn determinant_positive(self) -> bool {
        self != Self::II
    }

    /// Table 1's "type of e.p.", as printed.
    #[must_use]
    pub fn printed_type(self) -> &'static str {
        match self {
            Self::I => "stable node or spiral",
            Self::II => "unstable saddle",
            Self::III | Self::V => "stable focus or spiral",
            Self::IV => "unstable focus or spiral",
        }
    }
}

/// What an e.p. is, from the eigenvalues of its linearisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Two negative real eigenvalues.
    StableNode,
    /// Complex eigenvalues with negative real part.
    StableSpiral,
    /// Real eigenvalues of opposite sign: `Det < 0`.
    Saddle,
    /// Two positive real eigenvalues.
    UnstableNode,
    /// Complex eigenvalues with positive real part.
    UnstableSpiral,
    /// `Tr = 0` or `Det = 0`: the linearisation does not decide.
    NonHyperbolic,
}

/// The two-variable model, eqs. (13)–(14), which are (7)–(8) with a current `I`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoVariable {
    /// `a`, the cubic coefficient of `F(x) = a x³ − b x²`. Positive (p. 90).
    pub a: f64,
    /// `b`, the quadratic coefficient of `F`. Positive, with `b² > 3a` for region IV to exist.
    pub b: f64,
    /// `c`, the constant of `G(x) = c − d x²`. Positive.
    pub c: f64,
    /// `d`, the quadratic coefficient of `G`. Positive, with `b < d` for three e.p.s while `q > 0`
    /// (the paper's `c > 0` at `I = 0`).
    pub d: f64,
}

impl TwoVariable {
    /// The paper's `a = 1, b = 3, c = 1, d = 5`, used for every numerical result (Figs 3a, 4, 5, 6
    /// and 8 and p. 98): "for numerical investigation we will use `a = 1`, `b = 3`, `c = 1` and
    /// `d = 5` as previously" (p. 96).
    pub const PAPER: Self = Self { a: 1.0, b: 3.0, c: 1.0, d: 5.0 };

    /// The amplitude of the short current pulse of Figs 3a, 5a and 5c, "`I = 1`" in their captions.
    pub const PULSE_I: f64 = 1.0;

    /// A model with `a, b, c, d > 0`, the paper's condition on p. 90.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] naming the first of the four that is not finite and positive.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Result<Self, HindmarshRoseError> {
        let m = Self { a, b, c, d };
        m.check()?;
        Ok(m)
    }

    fn check(&self) -> Result<(), HindmarshRoseError> {
        positive("a", self.a)?;
        positive("b", self.b)?;
        positive("c", self.c)?;
        positive("d", self.d)?;
        Ok(())
    }

    /// The vector field `(ẋ, ẏ)` of (13)–(14) at `(x, y)` under current `i`.
    #[must_use]
    pub fn field(&self, x: f64, y: f64, i: f64) -> (f64, f64) {
        (y - self.a * x * x * x + self.b * x * x + i, self.c - self.d * x * x - y)
    }

    /// `p = (d − b)/a`, the quadratic coefficient of (9). The current does not enter it.
    #[must_use]
    pub fn p(&self) -> f64 {
        (self.d - self.b) / self.a
    }

    /// `q = (c + I)/a`, the right side of (9); the paper's `c/a` at `I = 0`.
    #[must_use]
    pub fn q(&self, i: f64) -> f64 {
        (self.c + i) / self.a
    }

    /// The discriminant of (9) as a cubic, `q(4p³ − 27q)`: three distinct e.p.s where it is
    /// positive, one where it is negative, and two (one of them double) where it is zero, unless
    /// `p = q = 0` and all three coincide.
    #[must_use]
    pub fn discriminant(&self, i: f64) -> f64 {
        let (p, q) = (self.p(), self.q(i));
        q * (4.0 * p * p * p - 27.0 * q)
    }

    /// The paper's condition for three e.p.s, `0 < 27q < 4p³` (p. 91; p. 92 writes it with the `0`).
    ///
    /// It is the `q > 0` half of `discriminant > 0`; see the module doc for the other half.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for a current that is not finite.
    pub fn three_point_condition(&self, i: f64) -> Result<bool, HindmarshRoseError> {
        self.check()?;
        finite("I", i)?;
        let (p, q) = (self.p(), self.q(i));
        Ok(0.0 < 27.0 * q && 27.0 * q < 4.0 * p * p * p)
    }

    /// The abscissae of the e.p.s under current `i`: the real roots of (9), ascending, each once.
    ///
    /// Their number follows the sign of [`TwoVariable::discriminant`] exactly — three, two or one —
    /// because the solver decides by that same number, computed the same way; at `q = 0`, the current
    /// `I = −c`, the double root at the origin is factored out exactly.
    ///
    /// ⚠ Two roots closer than about `√ε` of `|p|` are counted but not resolved: they come out at the
    /// same point. That happens near `q = 0`, where the small roots are `±√(q/p)`: for `p = 7/8` and
    /// `q = 2⁻⁶⁰` both land on `0` instead of `±1.0 × 10⁻⁹`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for a current that is not finite.
    pub fn equilibria(&self, i: f64) -> Result<Vec<f64>, HindmarshRoseError> {
        self.check()?;
        finite("I", i)?;
        Ok(cubic_roots(self.p(), 0.0, -self.q(i)))
    }

    /// The steady-state current–voltage curve drawn above Fig. 4, `I(∞) = a x³ + (d − b) x² − c`:
    /// the current at which `x` is an e.p. Its zeros are the e.p.s at `I = 0` (p. 95).
    #[must_use]
    pub fn steady_current(&self, x: f64) -> f64 {
        self.a * x * x * x + (self.d - self.b) * x * x - self.c
    }

    /// The linearisation `A(x₀)` of (7)–(8) at the e.p. with abscissa `x0`, as printed on p. 91:
    /// `[[−3a x₀² + 2b x₀, 1], [−2d x₀, −1]]`. The current does not enter it.
    #[must_use]
    pub fn linearisation(&self, x0: f64) -> Linearisation {
        Linearisation { jacobian: [[-3.0 * self.a * x0 * x0 + 2.0 * self.b * x0, 1.0], [-2.0 * self.d * x0, -1.0]] }
    }

    /// `Tr(A(x₀)) = −3a x₀² + 2b x₀ − 1`, the paper's formula (p. 91).
    #[must_use]
    pub fn trace(&self, x0: f64) -> f64 {
        -3.0 * self.a * x0 * x0 + 2.0 * self.b * x0 - 1.0
    }

    /// `Det(A(x₀)) = 3a x₀² + 2(d − b) x₀`, the paper's formula (p. 91) — which is `dI(∞)/dx`.
    #[must_use]
    pub fn determinant(&self, x0: f64) -> f64 {
        3.0 * self.a * x0 * x0 + 2.0 * (self.d - self.b) * x0
    }

    /// `D = (b² − 3a)^½`, the half-width of the band where the trace is positive, times `3a`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter that is not finite and positive;
    /// [`HindmarshRoseError::NoUnstableBand`] when `b² ≤ 3a`, where the trace is nowhere positive
    /// and region IV is empty (p. 91 has it negative everywhere for `b² < 3a`).
    pub fn band(&self) -> Result<f64, HindmarshRoseError> {
        self.check()?;
        let s = self.b * self.b - 3.0 * self.a;
        if s > 0.0 { Ok(s.sqrt()) } else { Err(HindmarshRoseError::NoUnstableBand { a: self.a, b: self.b }) }
    }

    /// `L = (b − D)/3a`, eq. (10): the lower edge of region IV.
    ///
    /// # Errors
    ///
    /// As [`TwoVariable::band`].
    pub fn l(&self) -> Result<f64, HindmarshRoseError> {
        Ok((self.b - self.band()?) / (3.0 * self.a))
    }

    /// `M = (b + D)/3a`, eq. (11): the upper edge of region IV.
    ///
    /// # Errors
    ///
    /// As [`TwoVariable::band`].
    pub fn m(&self) -> Result<f64, HindmarshRoseError> {
        Ok((self.b + self.band()?) / (3.0 * self.a))
    }

    /// `b < d`, which Table 1 presupposes: otherwise its saddle interval is not left of the origin.
    fn ordered(&self) -> Result<(), HindmarshRoseError> {
        if self.b < self.d { Ok(()) } else { Err(HindmarshRoseError::RegionsOutOfOrder { b: self.b, d: self.d }) }
    }

    /// Table 1's region of `x0`, or `None` on one of its four boundaries, where the table's strict
    /// inequalities place it in no region and `Tr` or `Det` vanishes.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NonFinite`] for an `x0` that is not finite;
    /// [`HindmarshRoseError::NotPositive`] for a parameter that is not finite and positive;
    /// [`HindmarshRoseError::NoUnstableBand`] for `b² ≤ 3a`;
    /// [`HindmarshRoseError::RegionsOutOfOrder`] for `b ≥ d`.
    pub fn region(&self, x0: f64) -> Result<Option<Region>, HindmarshRoseError> {
        finite("x0", x0)?;
        let (l, m) = (self.l()?, self.m()?);
        self.ordered()?;
        let saddle = -2.0 * (self.d - self.b) / (3.0 * self.a);
        Ok(if x0 < saddle {
            Some(Region::I)
        } else if saddle < x0 && x0 < 0.0 {
            Some(Region::II)
        } else if 0.0 < x0 && x0 < l {
            Some(Region::III)
        } else if l < x0 && x0 < m {
            Some(Region::IV)
        } else if m < x0 {
            Some(Region::V)
        } else {
            None
        })
    }

    /// Condition (12), `L³ + pL² < q < M³ + pM²`: the positive root of (9) lies in region IV.
    ///
    /// For `q > 0`, `x³ + px² − q` changes sign once on `x > 0` whatever `p` is, so there is exactly
    /// one positive root `C`, and as `L` and `M` are positive, (12) is `L < C < M`. The paper has
    /// `q > 0` from `c > 0` (p. 91), and a current below `−c` takes it away. Table 1 and §(c), whose
    /// region IV this is and whose three e.p.s need it (p. 91), presuppose `b < d`; then `p > 0`,
    /// `x³ + px²` is positive on `x > 0`, and a current that makes `q ≤ 0` leaves no positive root at
    /// all. Without `b < d` one can remain — `a = 1, b = 5, c = 1, d = 2` under `I = −1` has
    /// `q = 0` and (9) is `x²(x − 3)` — so `b ≥ d` is refused first, as [`TwoVariable::region`]
    /// refuses it.
    ///
    /// # Errors
    ///
    /// As [`TwoVariable::band`]; [`HindmarshRoseError::NonFinite`] for a current that is not finite;
    /// [`HindmarshRoseError::RegionsOutOfOrder`] for `b ≥ d`;
    /// [`HindmarshRoseError::NoPositiveRoot`] for a current that makes `q ≤ 0`.
    pub fn region_iv_condition(&self, i: f64) -> Result<bool, HindmarshRoseError> {
        finite("I", i)?;
        let (l, m, p, q) = (self.l()?, self.m()?, self.p(), self.q(i));
        self.ordered()?;
        if !(q > 0.0) {
            return Err(HindmarshRoseError::NoPositiveRoot { q });
        }
        Ok(l * l * l + p * l * l < q && q < m * m * m + p * m * m)
    }

    /// The type of the e.p. at `x0`, from the eigenvalues of `A(x₀)`: a saddle where `Det < 0`, and
    /// otherwise stable or unstable by the sign of `Tr` and a node or a spiral by the sign of
    /// `Tr² − 4 Det`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for an `x0` that is not finite.
    pub fn kind(&self, x0: f64) -> Result<Kind, HindmarshRoseError> {
        self.check()?;
        finite("x0", x0)?;
        let l = self.linearisation(x0);
        let (t, det) = (l.trace(), l.determinant());
        Ok(if det < 0.0 {
            Kind::Saddle
        } else if det == 0.0 || t == 0.0 {
            Kind::NonHyperbolic
        } else {
            match (t < 0.0, t * t < 4.0 * det) {
                (true, false) => Kind::StableNode,
                (true, true) => Kind::StableSpiral,
                (false, false) => Kind::UnstableNode,
                (false, true) => Kind::UnstableSpiral,
            }
        })
    }

    /// One fourth-order Runge–Kutta step of length `h` from `(x, y)` under constant current `i`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter or step that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for a state or current that is not finite.
    pub fn step(&self, x: f64, y: f64, i: f64, h: f64) -> Result<(f64, f64), HindmarshRoseError> {
        let run = self.simulate([x, y], &[(1, i)], h, 0.0)?;
        Ok((run.end[0], run.end[1]))
    }

    /// Integrate (13)–(14) from `start = [x, y]` through `protocol`, a list of `(steps, I)` segments
    /// each held for `steps` steps of length `h`, counting a spike at each upward crossing of
    /// `threshold` by `x`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter or step that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for a start, current or threshold that is not finite;
    /// [`HindmarshRoseError::Diverged`] if the state leaves the finite numbers.
    pub fn simulate(&self, start: [f64; 2], protocol: &[(usize, f64)], h: f64, threshold: f64) -> Result<Run<2>, HindmarshRoseError> {
        self.check()?;
        integrate(
            |[x, y], i| {
                let (dx, dy) = self.field(x, y, i);
                [dx, dy]
            },
            start,
            protocol,
            h,
            threshold,
        )
    }
}

/// The three-variable model with adaptation, eq. (15).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeVariable {
    /// The `x`–`y` system it adapts, (13)–(14).
    pub base: TwoVariable,
    /// `r`, the rate of the adaptation current, per time unit. "`z` is slowly varying
    /// (`r = 0.001`) compared to `x` and `y`" (p. 97).
    pub r: f64,
    /// `s`, the slope of the adaptation current's steady state `s(x − x₁)`, dimensionless.
    pub s: f64,
    /// `x₁`, the abscissa of the leftmost e.p. of `base` at `I = 0`, which makes `z = 0` its rest.
    pub x1: f64,
}

impl ThreeVariable {
    /// `x₁ = −(1 + √5)/2`, the exact leftmost root of (9) for [`TwoVariable::PAPER`] at `I = 0`,
    /// to the nearest double. The paper does not print it.
    pub const X1: f64 = -1.618_033_988_749_895;

    /// `−1.6`, the paper's rounding of the same e.p. (pp. 94 and 95). Not an equilibrium: see the
    /// module doc.
    pub const X1_PRINTED: f64 = -1.6;

    /// Fig. 5a: `r = 0.001, s = 1`, "an isolated burst of action potentials" after a short pulse.
    pub const FIG5A: Self = Self { base: TwoVariable::PAPER, r: 0.001, s: 1.0, x1: Self::X1 };

    /// Fig. 5c: `r = 0.001, s = 4`, a single spike and a depolarizing afterpotential.
    pub const FIG5C: Self = Self { base: TwoVariable::PAPER, r: 0.001, s: 4.0, x1: Self::X1 };

    /// Fig. 6: `r = 0.001, s = 4` under steady currents 0.4, 2 and 4 (Fig. 6 caption, p. 98).
    pub const FIG6: Self = Self { base: TwoVariable::PAPER, r: 0.001, s: 4.0, x1: Self::X1 };

    /// Fig. 8: "other parameters as in figures 5c and 6", under a hyperpolarizing step `I = −3`.
    pub const FIG8: Self = Self { base: TwoVariable::PAPER, r: 0.001, s: 4.0, x1: Self::X1 };

    /// The random burst structure of p. 98: `r = 0.005, s = 4` under `I = 3.25`.
    pub const RANDOM_BURSTS: Self = Self { base: TwoVariable::PAPER, r: 0.005, s: 4.0, x1: Self::X1 };

    /// The steady currents of Fig. 6a, 6b and 6c.
    pub const FIG6_I: [f64; 3] = [0.4, 2.0, 4.0];

    /// Fig. 8's hyperpolarizing current.
    pub const FIG8_I: f64 = -3.0;

    /// The random-burst current of p. 98.
    pub const RANDOM_BURSTS_I: f64 = 3.25;

    /// A model adapting `base` at rate `r` with slope `s`, with `x₁` the leftmost e.p. of `base` at
    /// `I = 0`, found by [`TwoVariable::equilibria`].
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter of `base`, an `r` or an `s` that is not
    /// finite and positive.
    pub fn new(base: TwoVariable, r: f64, s: f64) -> Result<Self, HindmarshRoseError> {
        positive("r", r)?;
        positive("s", s)?;
        let x1 = base.equilibria(0.0)?[0];
        Ok(Self { base, r, s, x1 })
    }

    fn check(&self) -> Result<(), HindmarshRoseError> {
        self.base.check()?;
        positive("r", self.r)?;
        positive("s", self.s)?;
        finite("x1", self.x1)?;
        Ok(())
    }

    /// The vector field `[ẋ, ẏ, ż]` of (15) at `u = [x, y, z]` under current `i`.
    #[must_use]
    pub fn field(&self, u: [f64; 3], i: f64) -> [f64; 3] {
        let [x, y, z] = u;
        let (dx, dy) = self.base.field(x, y, i);
        [dx - z, dy, self.r * (self.s * (x - self.x1) - z)]
    }

    /// `(x₁, y₁, 0)` with `y₁ = c − d x₁²`: the rest state, and an e.p. of (15) at `I = 0` exactly
    /// when `x₁` is a root of (9).
    #[must_use]
    pub fn rest(&self) -> [f64; 3] {
        [self.x1, self.base.c - self.base.d * self.x1 * self.x1, 0.0]
    }

    /// The e.p.s of (15) under current `i`, ascending in `x`.
    ///
    /// `ż = 0` gives `z = s(x − x₁)` and `ẏ = 0` gives `y = c − d x²`, so `ẋ = 0` reads
    /// `a x³ + (d − b) x² + s x − (c + I + s x₁) = 0` — (9) with a linear term. With the paper's
    /// constants and `s = 4` its slope `3x² + 4x + 4` never vanishes, so there is exactly one.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] or [`HindmarshRoseError::NonFinite`] for a parameter or
    /// current out of range.
    pub fn equilibria(&self, i: f64) -> Result<Vec<[f64; 3]>, HindmarshRoseError> {
        self.check()?;
        finite("I", i)?;
        let m = self.base;
        let roots = cubic_roots(m.p(), self.s / m.a, -(m.c + i + self.s * self.x1) / m.a);
        Ok(roots.into_iter().map(|x| [x, m.c - m.d * x * x, self.s * (x - self.x1)]).collect())
    }

    /// The Jacobian of (15) at abscissa `x`: `[[−3a x² + 2b x, 1, −1], [−2d x, −1, 0], [rs, 0, −r]]`.
    ///
    /// The paper does not derive it; only the stability of `(x₁, y₁, 0)` is stated (p. 96).
    #[must_use]
    pub fn jacobian(&self, x: f64) -> [[f64; 3]; 3] {
        let m = self.base;
        [[-3.0 * m.a * x * x + 2.0 * m.b * x, 1.0, -1.0], [-2.0 * m.d * x, -1.0, 0.0], [self.r * self.s, 0.0, -self.r]]
    }

    /// The characteristic polynomial `λ³ + c₂λ² + c₁λ + c₀` of [`ThreeVariable::jacobian`], as
    /// `[c₂, c₁, c₀]`: minus the trace, the sum of the principal 2×2 minors, minus the determinant.
    #[must_use]
    pub fn characteristic(&self, x: f64) -> [f64; 3] {
        let j = self.jacobian(x);
        let trace = j[0][0] + j[1][1] + j[2][2];
        let minors = (j[0][0] * j[1][1] - j[0][1] * j[1][0])
            + (j[0][0] * j[2][2] - j[0][2] * j[2][0])
            + (j[1][1] * j[2][2] - j[1][2] * j[2][1]);
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        [-trace, minors, -det]
    }

    /// Whether every eigenvalue of the Jacobian at `x` has a negative real part, by the
    /// Routh–Hurwitz conditions on the cubic: `c₂ > 0`, `c₀ > 0` and `c₂c₁ > c₀`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] or [`HindmarshRoseError::NonFinite`] for a parameter out
    /// of range; [`HindmarshRoseError::NonFinite`] for an `x` that is not finite.
    pub fn is_stable(&self, x: f64) -> Result<bool, HindmarshRoseError> {
        self.check()?;
        finite("x", x)?;
        let [c2, c1, c0] = self.characteristic(x);
        Ok(c2 > 0.0 && c0 > 0.0 && c2 * c1 > c0)
    }

    /// One fourth-order Runge–Kutta step of length `h` from `u` under constant current `i`.
    ///
    /// # Errors
    ///
    /// As [`ThreeVariable::simulate`].
    pub fn step(&self, u: [f64; 3], i: f64, h: f64) -> Result<[f64; 3], HindmarshRoseError> {
        Ok(self.simulate(u, &[(1, i)], h, 0.0)?.end)
    }

    /// Integrate (15) from `start = [x, y, z]` through `protocol`, a list of `(steps, I)` segments
    /// each held for `steps` steps of length `h`, counting a spike at each upward crossing of
    /// `threshold` by `x`.
    ///
    /// # Errors
    ///
    /// [`HindmarshRoseError::NotPositive`] for a parameter or step that is not finite and positive;
    /// [`HindmarshRoseError::NonFinite`] for an `x₁`, start, current or threshold that is not
    /// finite; [`HindmarshRoseError::Diverged`] if the state leaves the finite numbers.
    pub fn simulate(&self, start: [f64; 3], protocol: &[(usize, f64)], h: f64, threshold: f64) -> Result<Run<3>, HindmarshRoseError> {
        self.check()?;
        integrate(|u, i| self.field(u, i), start, protocol, h, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::{HindmarshRoseError, Kind, Region, ThreeVariable, TwoVariable, burst_sizes, cubic_roots};

    /// The step every simulation test uses, time units.
    const H: f64 = 0.01;

    fn steps(t: f64) -> usize {
        (t / H).round() as usize
    }

    /// The first spike of each burst, splitting where two spikes are more than `gap` apart.
    fn burst_starts(spikes: &[f64], gap: f64) -> Vec<f64> {
        let mut out = vec![spikes[0]];
        out.extend(spikes.windows(2).filter(|w| w[1] - w[0] > gap).map(|w| w[1]));
        out
    }

    fn worst(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "{a:?} against {b:?}");
        a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    // Every constant below is printed by `tools/hindmarshrose_reference.py`: `SciPy`'s `DOP853` at
    // rtol = atol = 10⁻¹² from the rest state `(x₁, y₁, 0)`, spikes as its events on `x = 1` upwards.

    /// Fig. 6a's spike times with the exact `x₁`; the same run under Radau agrees to within 4 × 10⁻¹⁰.
    const REF_6A: [f64; 8] = [
        42.66863897889993,
        53.10791834116552,
        64.23103337788407,
        76.21863367948654,
        89.3589900937369,
        104.17571561316191,
        121.87734103593922,
        148.01926882630258,
    ];
    /// Fig. 6a's spike times with `x₁ = −1.6`, from `(−1.6, −11.8, 0)`; Radau agrees to within 2.8 × 10⁻¹⁰.
    const REF_6A_PRINTED: [f64; 8] = [
        41.58215406901368,
        51.97227708331474,
        63.0231970498407,
        74.90270777177416,
        87.87362778028269,
        102.39898383246145,
        119.48405610081879,
        142.80005665762587,
    ];
    /// Fig. 6b's burst starts (a gap of 50) over 3 000 time units; Radau agrees to within 7 × 10⁻⁹.
    const REF_6B: [f64; 6] =
        [6.677539611441476, 852.5577480779766, 1305.3993621825375, 1758.2414258350063, 2211.0834894528098, 2663.9255530706373];
    /// The same with `x₁ = −1.6`; the seventh burst is cut off at 3 000. Radau agrees to within 2.1 × 10⁻⁹.
    const REF_6B_PRINTED: [f64; 7] = [
        6.512503693493694,
        837.4238370459977,
        1268.1990004956556,
        1698.974612291501,
        2129.7502240199938,
        2560.5258357484845,
        2991.30144747698,
    ];
    /// The first 20 spike times of the random-burst run of p. 98.
    const REF_RANDOM_20: [f64; 20] = [
        3.951747795988624,
        7.389275107583243,
        10.903036396953729,
        14.49744135652723,
        18.177307247172823,
        21.947914227071017,
        25.81506930671117,
        29.785182063061747,
        33.86535486351562,
        38.06349114610067,
        42.38842637692776,
        46.85008777514691,
        51.4596909332352,
        56.229984329561184,
        61.17555683913259,
        66.31322934364411,
        71.66256047274494,
        77.2465101245389,
        83.09232572246708,
        89.23275051896861,
    ];
    /// The random-burst run's burst sizes over 6 000 time units, split at a gap of 60.
    const REF_RANDOM_SIZES: [usize; 36] =
        [31, 5, 6, 5, 5, 3, 5, 5, 3, 5, 5, 5, 5, 4, 4, 7, 7, 5, 2, 5, 7, 5, 5, 5, 3, 7, 7, 7, 7, 5, 4, 5, 4, 3, 5, 7];

    /// The paper's cubic at `I = 0` is `x³ + 2x² − 1 = (x + 1)(x² + x − 1)`, and its three roots come
    /// out bit for bit on `−(1 + √5)/2`, `−1` and `(√5 − 1)/2`, evaluated in `f64` — which are the
    /// NEAREST doubles to the exact roots.
    ///
    /// Nearest, exactly: `g(x) = x² + x − 1` changes sign between the two points halfway to each
    /// root's neighbouring doubles. Those points are not doubles, so `g` is evaluated in integers —
    /// a point `k·2^E` with `k` odd has `2^(−2E) g = k² + k·2^(−E) − 2^(−2E)`, which fits an `i128`.
    /// The paper prints the roots rounded, "−1.6, −1 and +0.6" (p. 94), and to one decimal they are
    /// exactly that. `numpy.roots` on the same cubic returns `−1.6180339887498936`,
    /// `−1.0000000000000007` and `0.6180339887498948`, within a measured 1.3 × 10⁻¹⁵ of these. `y₁`
    /// computed as `c − d x₁²` equals `−(13 + 5√5)/2` computed directly, to the bit here; the bound
    /// allows a few ulps.
    #[test]
    fn the_papers_cubic_has_the_golden_ratio_for_roots() {
        let m = TwoVariable::PAPER;
        let r5 = 5.0_f64.sqrt();
        let e = m.equilibria(0.0).unwrap();
        assert_eq!(e, vec![-(1.0 + r5) / 2.0, -1.0, (r5 - 1.0) / 2.0]);
        assert_eq!(e[0], ThreeVariable::X1);
        assert_eq!((m.p(), m.q(0.0)), (2.0, 1.0));
        let printed: Vec<f64> = e.iter().map(|x| (x * 10.0).round() / 10.0).collect();
        assert_eq!(printed, vec![-1.6, -1.0, 0.6]);
        let numpy = [-1.6180339887498936, -1.0000000000000007, 0.6180339887498948];
        let d = worst(&e, &numpy);
        assert!(d < 4e-15, "{d}");
        let [x1, y1, z1] = ThreeVariable::FIG6.rest();
        assert!((y1 + (13.0 + 5.0 * r5) / 2.0).abs() < 4e-15, "{y1}");
        assert_eq!((x1, z1), (ThreeVariable::X1, 0.0));
        for x in [e[0], e[2]] {
            // `|x| = m·2^(b − 1075)` for a normal double with biased exponent `b`; the midpoints to
            // its neighbours are `(2m ± 1)·2^(b − 1076)`, carrying the sign of `x`.
            let bits = x.abs().to_bits();
            let (m, shift) = (i128::from(bits & ((1 << 52) - 1) | (1 << 52)), 1076 - (bits >> 52) as u32);
            let sign = |k: i128| -> i128 {
                let k = if x < 0.0 { -k } else { k };
                (k * k + k * (1_i128 << shift) - (1_i128 << (2 * shift))).signum()
            };
            assert_eq!(sign(2 * m - 1) * sign(2 * m + 1), -1, "{x} is not the nearest double to a root of x² + x − 1");
            assert_eq!(sign(2 * m + 1) * sign(2 * m + 3), 1, "and the test can tell: one double over, it is not");
        }
    }

    /// With a current, (9) keeps `p` and raises `q` to `(c + I)/a`, and every root is a zero of
    /// (13)–(14) on the parabola `y = c − d x²` and a point where `I(∞)` equals the current.
    ///
    /// Swept over currents from −3 to 4 in steps of 1/8, which crosses both ends of the three-root
    /// window `(I(∞)(0), I(∞)(−4/3)) = (−1, 5/27)`: three e.p.s inside it, one outside, and at its
    /// lower end `I = −1` two, the simple one at −2 and the double one at the origin, where `q = 0` and
    /// `x³ + 2x² = x²(x + 2)`. The largest field residual at a root measures 3.6 × 10⁻¹⁵, and
    /// `I(∞) − I` 1.8 × 10⁻¹⁵. At Fig. 6b's `I = 2` the one e.p. is `(1, −4)` exactly: `q = 3` and
    /// `x³ + 2x² − 3 = (x − 1)(x² + 3x + 3)`.
    #[test]
    fn the_current_raises_q_and_every_root_is_a_zero_of_the_field() {
        let m = TwoVariable::PAPER;
        assert!((m.steady_current(-4.0 / 3.0) - 5.0 / 27.0).abs() < 1e-15 && m.steady_current(0.0) == -1.0);
        let (mut field, mut iinf) = (0.0_f64, 0.0_f64);
        for n in -24..=32 {
            let i = f64::from(n) / 8.0;
            assert_eq!((m.p(), m.q(i)), (2.0, 1.0 + i));
            let roots = m.equilibria(i).unwrap();
            let count = if i == -1.0 { 2 } else if -1.0 < i && i < 5.0 / 27.0 { 3 } else { 1 };
            assert_eq!(roots.len(), count, "I = {i}: {roots:?}");
            for &x in &roots {
                let (dx, dy) = m.field(x, m.c - m.d * x * x, i);
                field = field.max(dx.abs()).max(dy.abs());
                iinf = iinf.max((m.steady_current(x) - i).abs());
            }
        }
        assert!(field < 1e-14 && iinf < 5e-15, "{field} {iinf}");
        assert_eq!(m.equilibria(-1.0).unwrap(), vec![-2.0, 0.0]);
        assert_eq!(m.equilibria(2.0).unwrap(), vec![1.0]);
        assert_eq!(m.field(1.0, -4.0, 2.0), (0.0, 0.0));
    }

    /// The count of e.p.s is the sign of the discriminant `q(4p³ − 27q)` — three where it is
    /// positive, one where it is negative, two where it is zero — and the paper's condition
    /// `0 < 27q < 4p³` is its positive-`q` half.
    ///
    /// Two sweeps, with no point skipped. Over `a`, `b`, `d` and the current on binary fractions, all
    /// four outcomes occur: three roots with `q > 0`, three with `q < 0`, one, and two. And close to
    /// the paper's own boundary `q = 0`, where rounding is hardest: `p = k/8` from −19.875 to 20 and
    /// `q = ±3·2⁻ʲ` for `j` from 0 to 64, 41 600 points, with `c = |q|` so that `q` is exact. There
    /// every count follows the sign, and the field at every root is at most a measured 2.5 × 10⁻¹⁵
    /// times `1 + |x|³ + |p|³`, the size of the terms (9) balances: each root is an exact e.p. under
    /// a current moved by that much, the unresolved pairs near the origin included (see
    /// [`TwoVariable::equilibria`]). On the other boundary, exact cases: `p = 3k`, `q = 4k³` for
    /// `k = j/8` up to 20 gives `x³ + 3k x² − 4k³ = (x − k)(x + 2k)²`, a discriminant of exactly zero
    /// and the two e.p.s `−2k` and `k` exactly, where the paper's strict condition is false. `p = q = 0`
    /// gives one triple root at the origin. Off the paper's half, `a = 1, b = 5, c = 1, d = 2` under
    /// `I = −3` has `p = −3, q = −2` and three e.p.s, `1 − √3`, `1` and `1 + √3` (to a measured
    /// 1.1 × 10⁻¹⁶), which `0 < 27q` rejects. The paper's own constants under Fig. 8's `I = −3` have
    /// `q = −2` and one e.p.
    #[test]
    fn the_count_is_the_discriminant_and_the_papers_condition_is_its_positive_half() {
        let want = |disc: f64, p: f64, q: f64| -> usize {
            if disc > 0.0 {
                3
            } else if disc < 0.0 || (p == 0.0 && q == 0.0) {
                1
            } else {
                2
            }
        };
        let mut seen = [0; 4];
        for a in [0.5, 1.0, 2.0] {
            for b in [1.0, 3.0, 6.0] {
                for d in [0.5, 2.0, 5.0, 9.0] {
                    for n in -40..=40 {
                        let m = TwoVariable { a, b, c: 1.0, d };
                        let i = f64::from(n) / 8.0;
                        let (disc, count) = (m.discriminant(i), m.equilibria(i).unwrap().len());
                        assert_eq!(count, want(disc, m.p(), m.q(i)), "{m:?} at I = {i}");
                        if m.q(i) > 0.0 {
                            assert_eq!(m.three_point_condition(i).unwrap(), count == 3, "{m:?} at I = {i}");
                        } else {
                            assert!(!m.three_point_condition(i).unwrap());
                        }
                        match (count, m.q(i) > 0.0) {
                            (3, true) => seen[0] += 1,
                            (3, false) => seen[1] += 1,
                            (1, _) => seen[2] += 1,
                            _ => seen[3] += 1,
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
        let (mut near, mut residual) = ([0; 4], 0.0_f64);
        for k in -159..=160 {
            for j in 0..=64 {
                for below in [false, true] {
                    let size = 3.0 * 2.0_f64.powi(-j);
                    let (m, i) = (TwoVariable { a: 1.0, b: 21.0, c: size, d: 21.0 + f64::from(k) / 8.0 }, if below { -2.0 * size } else { 0.0 });
                    assert_eq!((m.p(), m.q(i)), (f64::from(k) / 8.0, if below { -size } else { size }));
                    let roots = m.equilibria(i).unwrap();
                    let count = want(m.discriminant(i), m.p(), m.q(i));
                    assert_eq!(roots.len(), count, "p = {}, q = {}: {roots:?}", m.p(), m.q(i));
                    near[count] += 1;
                    for &x in &roots {
                        let (dx, dy) = m.field(x, m.c - m.d * x * x, i);
                        residual = residual.max(dx.abs().max(dy.abs()) / (1.0 + x.abs().powi(3) + m.p().abs().powi(3)));
                    }
                }
            }
        }
        assert_eq!(near, [0, 21_053, 0, 20_547]);
        assert!(residual < 1e-14, "{residual}");
        for j in 1..=160 {
            let k = f64::from(j) / 8.0;
            let m = TwoVariable { a: 1.0, b: 1.0, c: 1.0, d: 1.0 + 3.0 * k };
            let i = 4.0 * k * k * k - 1.0;
            assert_eq!((m.p(), m.q(i), m.discriminant(i)), (3.0 * k, 4.0 * k * k * k, 0.0));
            assert_eq!(m.equilibria(i).unwrap(), vec![-2.0 * k, k], "k = {k}");
            assert!(!m.three_point_condition(i).unwrap(), "27q = 4p³ is not below it");
        }
        let double = TwoVariable { a: 1.0, b: 3.0, c: 1.0, d: 6.0 };
        assert!(double.three_point_condition(3.0 - 1.0 / 64.0).unwrap() && double.equilibria(3.0 - 1.0 / 64.0).unwrap().len() == 3);
        assert!(!double.three_point_condition(3.0 + 1.0 / 64.0).unwrap() && double.equilibria(3.0 + 1.0 / 64.0).unwrap().len() == 1);
        let triple = TwoVariable { a: 1.0, b: 3.0, c: 1.0, d: 3.0 };
        assert_eq!((triple.discriminant(-1.0), triple.equilibria(-1.0).unwrap()), (0.0, vec![0.0]));
        assert!(!TwoVariable::PAPER.three_point_condition(-1.0).unwrap(), "q = 0 is not above zero");
        let flipped = TwoVariable { a: 1.0, b: 5.0, c: 1.0, d: 2.0 };
        assert_eq!((flipped.p(), flipped.q(-3.0), flipped.discriminant(-3.0)), (-3.0, -2.0, 108.0));
        let r3 = 3.0_f64.sqrt();
        let got = flipped.equilibria(-3.0).unwrap();
        assert!(worst(&got, &[1.0 - r3, 1.0, 1.0 + r3]) < 5e-16, "{got:?}");
        assert!(!flipped.three_point_condition(-3.0).unwrap());
        let m = TwoVariable::PAPER;
        assert_eq!((m.q(ThreeVariable::FIG8_I), m.equilibria(ThreeVariable::FIG8_I).unwrap().len()), (-2.0, 1));
        assert!(m.discriminant(ThreeVariable::FIG8_I) < 0.0);
    }

    /// At `q = 0` — the current `I = −c`, the edge of the paper's `0 < 27q` — (9) is `x²(x + p) = 0`:
    /// a double e.p. at the origin beside a simple one at `−p`, and the solver returns exactly those,
    /// the origin once and as `+0`, for all 321 of `p = k/8` from −20 to 20 (the origin alone at
    /// `p = 0`), where [`TwoVariable::discriminant`] is exactly zero; the field there is exactly zero.
    ///
    /// The depressed discriminant the solver used to decide by, `−(4P³ + 27Q²)` with `P = −p²/3` and
    /// `Q = 2p³/27` as it computed them, is `−7.3 × 10⁻¹²` at `p = 7`, where it took Cardano's branch
    /// and returned only `−7`, and `+1.1 × 10⁻¹⁶` at `p = 5/4`, where it took the three-root branch
    /// and split the origin in two.
    #[test]
    fn at_q_zero_the_origin_is_a_double_root_found_exactly() {
        for k in -160..=160 {
            let p = f64::from(k) / 8.0;
            let m = TwoVariable { a: 1.0, b: 21.0, c: 1.0, d: 21.0 + p };
            assert_eq!((m.p(), m.q(-1.0), m.discriminant(-1.0)), (p, 0.0, 0.0));
            let roots = m.equilibria(-1.0).unwrap();
            let want = if k == 0 { vec![0.0] } else if k > 0 { vec![-p, 0.0] } else { vec![0.0, -p] };
            assert_eq!(roots, want, "p = {p}");
            assert!(roots.iter().all(|x| x.is_sign_positive() || *x != 0.0), "{roots:?}");
            assert_eq!(m.field(0.0, m.c, -1.0), (0.0, 0.0));
        }
        assert_eq!(TwoVariable::PAPER.equilibria(-1.0).unwrap(), vec![-2.0, 0.0]);
        let former = |p: f64| {
            let (big_p, big_q) = (-(p * p) / 3.0, 2.0 * p * p * p / 27.0);
            -(4.0 * big_p * big_p * big_p + 27.0 * big_q * big_q)
        };
        assert!((former(7.0) + 7.3e-12).abs() < 5e-14 && (former(1.25) - 1.1e-16).abs() < 5e-18, "{} {}", former(7.0), former(1.25));
    }

    /// The printed `Tr` and `Det` are the trace and determinant of the printed `A(x₀)`, `A(x₀)` is the
    /// derivative of the field, and `Det` is the slope of `I(∞)`.
    ///
    /// Central differences with step 10⁻⁶ at points off both nullclines, for the paper's constants
    /// and for `a = 0.5, b = 2, c = 3, d = 7`, where no coefficient is one and none coincides: the
    /// worst Jacobian entry disagrees by a measured 1.5 × 10⁻¹⁰ of the largest, `dI(∞)/dx` differs from
    /// `Det` by 1.7 × 10⁻¹⁰ of its size, and the printed formulas match the matrix's trace and
    /// determinant to 3.6 × 10⁻¹⁵.
    #[test]
    fn the_printed_trace_and_determinant_are_the_matrix_and_the_matrix_is_the_derivative() {
        let (mut jac, mut slope, mut formula) = (0.0_f64, 0.0_f64, 0.0_f64);
        for m in [TwoVariable::PAPER, TwoVariable { a: 0.5, b: 2.0, c: 3.0, d: 7.0 }] {
            for (x, y, i) in [(0.3, -0.4, 0.0), (-1.7, 0.9, 0.5), (2.2, 1.5, -1.25), (-0.6, -8.0, 3.0)] {
                let l = m.linearisation(x);
                let e = 1e-6;
                let (a, b) = (m.field(x + e, y, i), m.field(x - e, y, i));
                let (c, d) = (m.field(x, y + e, i), m.field(x, y - e, i));
                let fd = [[(a.0 - b.0) / (2.0 * e), (c.0 - d.0) / (2.0 * e)], [(a.1 - b.1) / (2.0 * e), (c.1 - d.1) / (2.0 * e)]];
                let scale = l.jacobian.iter().flatten().fold(0.0_f64, |s, v| s.max(v.abs()));
                for (p, q) in l.jacobian.iter().flatten().zip(fd.iter().flatten()) {
                    jac = jac.max((p - q).abs() / scale);
                }
                formula = formula.max((l.trace() - m.trace(x)).abs()).max((l.determinant() - m.determinant(x)).abs());
                let ds = (m.steady_current(x + e) - m.steady_current(x - e)) / (2.0 * e);
                slope = slope.max((ds - m.determinant(x)).abs() / m.determinant(x).abs().max(1.0));
            }
        }
        assert!(jac < 5e-10 && slope < 5e-10 && formula < 1e-14, "{jac} {slope} {formula}");
    }

    /// Table 1, region by region: across `x₀` from −3 to 3 the signs of `Tr` and `Det` are the ones
    /// the table prints for the region, the regions come in the printed order, and each of the four
    /// boundaries belongs to none.
    ///
    /// For the paper's constants `D = √6`, `L = (3 − √6)/3 = 0.183503`, `M = (3 + √6)/3 = 1.816497`,
    /// and the saddle interval starts at `−2(d − b)/3a = −4/3`. Where no coefficient is one, the
    /// boundaries are still the zeros the table says they are — `L` and `M` the zeros of `Tr`,
    /// `−2(d − b)/3a` the negative zero of `Det` — to a measured 4.4 × 10⁻¹⁶.
    #[test]
    fn table_1_region_by_region() {
        let m = TwoVariable::PAPER;
        let r6 = 6.0_f64.sqrt();
        assert_eq!(m.band().unwrap(), r6);
        assert!((m.l().unwrap() - (3.0 - r6) / 3.0).abs() < 1e-16 && (m.m().unwrap() - (3.0 + r6) / 3.0).abs() < 1e-15);
        assert!((m.l().unwrap() - 0.183503).abs() < 5e-7 && (m.m().unwrap() - 1.816497).abs() < 5e-7);
        let mut order = Vec::new();
        for n in -192..=192 {
            let x0 = f64::from(n) / 64.0;
            if let Some(r) = m.region(x0).unwrap() {
                assert_eq!(m.trace(x0) > 0.0, r.trace_positive(), "{r:?} at {x0}");
                assert_eq!(m.determinant(x0) > 0.0, r.determinant_positive(), "{r:?} at {x0}");
                if order.last() != Some(&r) {
                    order.push(r);
                }
            } else {
                assert_eq!(x0, 0.0, "the only boundary on this grid is the origin");
            }
        }
        assert_eq!(order, vec![Region::I, Region::II, Region::III, Region::IV, Region::V]);
        for edge in [-4.0 / 3.0, 0.0, m.l().unwrap(), m.m().unwrap()] {
            assert_eq!(m.region(edge).unwrap(), None, "{edge}");
        }
        let printed: Vec<&str> = order.iter().map(|r| r.printed_type()).collect();
        assert_eq!(
            printed,
            vec!["stable node or spiral", "unstable saddle", "stable focus or spiral", "unstable focus or spiral", "stable focus or spiral"]
        );
        // `a = 1, b = 2` puts M at exactly 1, and just past it is region V.
        let unit = TwoVariable { a: 1.0, b: 2.0, c: 1.0, d: 5.0 };
        assert_eq!(unit.m().unwrap(), 1.0);
        assert_eq!((unit.region(1.0).unwrap(), unit.region(1.0_f64.next_up()).unwrap()), (None, Some(Region::V)));
        assert_eq!(unit.region(1.0_f64.next_down()).unwrap(), Some(Region::IV));
        assert_eq!(unit.region(-2.0).unwrap(), None, "−2(d − b)/3a = −2");
        assert_eq!(unit.region((-2.0_f64).next_down()).unwrap(), Some(Region::I));
        assert_eq!(unit.region((-2.0_f64).next_up()).unwrap(), Some(Region::II));
        assert_eq!(unit.region(-f64::MIN_POSITIVE).unwrap(), Some(Region::II));
        assert_eq!(unit.region(f64::MIN_POSITIVE).unwrap(), Some(Region::III));
        let mut zero = 0.0_f64;
        for m in [TwoVariable { a: 2.0, b: 3.0, c: 1.0, d: 5.0 }, TwoVariable { a: 0.5, b: 2.0, c: 3.0, d: 7.0 }] {
            let (l, mm) = (m.l().unwrap(), m.m().unwrap());
            let edge = -2.0 * (m.d - m.b) / (3.0 * m.a);
            zero = zero.max(m.trace(l).abs()).max(m.trace(mm).abs()).max(m.determinant(edge).abs());
            assert_eq!(m.region(edge).unwrap(), None);
            assert_eq!((m.region(edge.next_down()).unwrap(), m.region(edge.next_up()).unwrap()), (Some(Region::I), Some(Region::II)));
            assert_eq!((m.region(l.next_down()).unwrap(), m.region(l.next_up()).unwrap()), (Some(Region::III), Some(Region::IV)));
            assert_eq!((m.region(mm.next_down()).unwrap(), m.region(mm.next_up()).unwrap()), (Some(Region::IV), Some(Region::V)));
        }
        assert!(zero < 2e-15, "{zero}");
    }

    /// Condition (12) holds exactly when the positive root of (9) lies strictly between `L` and `M`,
    /// and is refused where Table 1's `b < d` fails or there is no positive root.
    ///
    /// Swept over models with `b² > 3a` and `b < d`, and currents with `q > 0`, where (9) has exactly
    /// one positive root, its largest: both verdicts occur, and they always agree. With `b < d` and
    /// currents taking `q` from 0 down to −3 in eighths, no root of (9) is positive and every one is
    /// refused as having none; every model of the sweep with `b ≥ d` is refused as out of order at
    /// every current. That refusal comes first: `a = 1, b = 5, c = 1, d = 2` has `q = 0` under
    /// `I = −1` and still a positive root, (9) being `x²(x − 3)`, and under `I = −5` a double one,
    /// `(x + 1)(x − 2)²` — both found exactly. Each side is strict, checked where it is exact:
    /// `a = 1, b = 19/8` gives `D = 13/8` and `L = 1/4`, and with `p = 1` the left side is `5/64`, so
    /// `c = 5/64` puts the root at `L` itself; `a = 1, b = 2, d = 3` gives `M = 1` and a right side of
    /// 2. For the paper, `L³ + pL² = 0.073526 < q = 1 < M³ + pM² = 12.593140`.
    #[test]
    fn condition_12_is_the_rightmost_root_inside_region_iv() {
        let mut seen = (0, 0, 0, 0);
        for a in [0.5, 1.0, 2.0] {
            for b in [3.0, 4.0, 6.0] {
                for d in [2.0, 5.0, 9.0] {
                    let m = TwoVariable { a, b, c: 1.0, d };
                    if b >= d {
                        for n in -40..=80 {
                            let e = m.region_iv_condition(f64::from(n) / 8.0).unwrap_err();
                            assert_eq!(e, HindmarshRoseError::RegionsOutOfOrder { b, d });
                            seen.2 += 1;
                        }
                        continue;
                    }
                    for k in 0..=24 {
                        let i = -m.c - f64::from(k) / 8.0 * a;
                        assert!(m.q(i) <= 0.0 && m.equilibria(i).unwrap().iter().all(|&x| x <= 0.0), "{m:?} at I = {i}");
                        assert_eq!(m.region_iv_condition(i).unwrap_err(), HindmarshRoseError::NoPositiveRoot { q: m.q(i) });
                        seen.3 += 1;
                    }
                    for n in -7..=80 {
                        let i = f64::from(n) / 8.0;
                        let c_root = *m.equilibria(i).unwrap().last().unwrap();
                        let (l, mm) = (m.l().unwrap(), m.m().unwrap());
                        if (c_root - l).abs() < 1e-9 || (c_root - mm).abs() < 1e-9 {
                            continue;
                        }
                        let inside = l < c_root && c_root < mm;
                        assert_eq!(m.region_iv_condition(i).unwrap(), inside, "{m:?} at I = {i}: C = {c_root}");
                        if inside {
                            seen.0 += 1;
                        } else {
                            seen.1 += 1;
                        }
                    }
                }
            }
        }
        assert!(seen.0 > 0 && seen.1 > 0 && seen.2 > 0 && seen.3 > 0, "{seen:?}");
        let flipped = TwoVariable { a: 1.0, b: 5.0, c: 1.0, d: 2.0 };
        assert_eq!((flipped.q(-1.0), flipped.equilibria(-1.0).unwrap()), (0.0, vec![0.0, 3.0]));
        assert_eq!((flipped.q(-5.0), flipped.equilibria(-5.0).unwrap()), (-4.0, vec![-1.0, 2.0]));
        for i in [-1.0, -3.0, -5.0] {
            assert_eq!(flipped.region_iv_condition(i).unwrap_err(), HindmarshRoseError::RegionsOutOfOrder { b: 5.0, d: 2.0 });
        }
        let m = TwoVariable::PAPER;
        let (l, mm) = (m.l().unwrap(), m.m().unwrap());
        assert!((l * l * l + 2.0 * l * l - 0.073526).abs() < 5e-7 && (mm * mm * mm + 2.0 * mm * mm - 12.593140).abs() < 5e-7);
        assert!(m.region_iv_condition(0.0).unwrap());
        let low = TwoVariable { a: 1.0, b: 2.375, c: 5.0 / 64.0, d: 3.375 };
        assert_eq!((low.band().unwrap(), low.l().unwrap(), low.p()), (1.625, 0.25, 1.0));
        assert!(!low.region_iv_condition(0.0).unwrap(), "q = L³ + pL² is not above it");
        assert!(low.region_iv_condition(1.0 / 1024.0).unwrap());
        let high = TwoVariable { a: 1.0, b: 2.0, c: 2.0, d: 3.0 };
        assert_eq!((high.m().unwrap(), high.p()), (1.0, 1.0));
        assert!(!high.region_iv_condition(0.0).unwrap(), "q = M³ + pM² is not below it");
        assert!(high.region_iv_condition(-1.0 / 1024.0).unwrap());
    }

    /// Fig. 4: A is a stable node, B a saddle and C an unstable spiral, in regions I, II and IV —
    /// "as predicted by the analysis (table 1)" (p. 94).
    ///
    /// At a root of `x² + x − 1`, `x² = 1 − x` reduces the p. 91 formulas to `Tr = 9x − 4` and
    /// `Det = 3 + x`: `Tr = −18.562306, Det = 1.381966` at A and `1.562306, 3.618034` at C; at
    /// `B = −1`, `Tr = −10, Det = −1`.
    #[test]
    fn figure_4_is_a_stable_node_a_saddle_and_an_unstable_spiral() {
        let m = TwoVariable::PAPER;
        let e = m.equilibria(0.0).unwrap();
        let kinds: Vec<Kind> = e.iter().map(|&x| m.kind(x).unwrap()).collect();
        assert_eq!(kinds, vec![Kind::StableNode, Kind::Saddle, Kind::UnstableSpiral]);
        let regions: Vec<Option<Region>> = e.iter().map(|&x| m.region(x).unwrap()).collect();
        assert_eq!(regions, vec![Some(Region::I), Some(Region::II), Some(Region::IV)]);
        for x in [e[0], e[2]] {
            assert!((m.trace(x) - (9.0 * x - 4.0)).abs() < 1e-14 && (m.determinant(x) - (3.0 + x)).abs() < 1e-14, "{x}");
        }
        assert!((m.trace(e[0]) + 18.562306).abs() < 5e-7 && (m.determinant(e[0]) - 1.381966).abs() < 5e-7);
        assert!((m.trace(e[2]) - 1.562306).abs() < 5e-7 && (m.determinant(e[2]) - 3.618034).abs() < 5e-7);
        assert_eq!((m.trace(-1.0), m.determinant(-1.0)), (-10.0, -1.0));
    }

    /// Every branch of the classification, each at a point where it is exact.
    ///
    /// `Det(A(0)) = 0` for every model; `a = 1, b = 2` puts `Tr` at exactly zero at `x₀ = 1` with
    /// `Det = 9`, eigenvalues `±3i`, a Hopf point; `a = 1, b = 3, d = 2` gives `Tr = 2`, `Det = 1` at
    /// `x₀ = 1`, the repeated eigenvalue 1, a node on the edge of becoming a spiral; `x₀ = 0.1` is a
    /// stable spiral in the paper's region III, `x₀ = 1` an unstable spiral in its region IV and
    /// `x₀ = 2.5` a stable spiral in its region V; `a = 1, b = 10, d = 10.5` at `x₀ = 10/3` has
    /// `Tr = 32.3`, `Det = 36.7`, an unstable node; and `a = 1, b = 3.5, d = 4` at `x₀ = 1` has
    /// `Tr = 3`, `Det = 4`, so `Tr² = 9` lies between `2 Det` and `4 Det`: eigenvalues
    /// `(3 ± i√7)/2`, an unstable spiral.
    #[test]
    fn every_kind_of_equilibrium_is_classified() {
        let m = TwoVariable::PAPER;
        assert_eq!(m.kind(0.0).unwrap(), Kind::NonHyperbolic);
        let hopf = TwoVariable { a: 1.0, b: 2.0, c: 1.0, d: 5.0 };
        assert_eq!((hopf.trace(1.0), hopf.determinant(1.0), hopf.kind(1.0).unwrap()), (0.0, 9.0, Kind::NonHyperbolic));
        let edge = TwoVariable { a: 1.0, b: 3.0, c: 1.0, d: 2.0 };
        assert_eq!((edge.trace(1.0), edge.determinant(1.0), edge.kind(1.0).unwrap()), (2.0, 1.0, Kind::UnstableNode));
        assert_eq!((m.kind(0.1).unwrap(), m.kind(1.0).unwrap(), m.kind(2.5).unwrap()), (Kind::StableSpiral, Kind::UnstableSpiral, Kind::StableSpiral));
        assert_eq!(TwoVariable { a: 1.0, b: 10.0, c: 1.0, d: 10.5 }.kind(10.0 / 3.0).unwrap(), Kind::UnstableNode);
        let wide = TwoVariable { a: 1.0, b: 3.5, c: 1.0, d: 4.0 };
        assert_eq!((wide.trace(1.0), wide.determinant(1.0), wide.kind(1.0).unwrap()), (3.0, 4.0, Kind::UnstableSpiral));
    }

    /// `(x₁, y₁, 0)` is an e.p. of (15) at `I = 0`, and the printed `(−1.6, −11.8, 0)` is not.
    ///
    /// At the exact root the field is `[0, 0, 0]` — exactly, every term cancelling in floating point.
    /// At the rounded point `ẋ = G(−1.6) − F(−1.6) = 1 − 12.8 + 4.096 + 7.68 = −0.024` (measured
    /// −0.023999999999998245, the rest being −1.6's own rounding), with `ẏ = ż = 0`. And a model run
    /// with `x₁ = −1.6` rests somewhere else again: for `s = 4` at `x = −1.6045345328021472` with an
    /// adaptation current `z = −0.018` that is not zero, against `numpy.roots`' `−1.604534532802149`,
    /// and for Fig. 5a's `s = 1` at `x = −1.6103932`, `z = −0.0104`, against `−1.6103931697431384`.
    #[test]
    fn the_rest_state_is_exact_at_the_root_and_not_at_the_printed_rounding() {
        for f in [ThreeVariable::FIG5A, ThreeVariable::FIG6, ThreeVariable::RANDOM_BURSTS] {
            assert_eq!(f.field(f.rest(), 0.0), [0.0; 3]);
        }
        let printed = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..ThreeVariable::FIG6 };
        let [dx, dy, dz] = printed.field([-1.6, 1.0 - 5.0 * 2.56, 0.0], 0.0);
        assert!((dx + 0.024).abs() < 1e-14 && dy == 0.0 && dz == 0.0, "{dx} {dy} {dz}");
        let rest = printed.equilibria(0.0).unwrap();
        assert_eq!(rest.len(), 1);
        let [x, y, z] = rest[0];
        assert!((x - -1.604534532802149).abs() < 5e-15 && (z - 4.0 * (x + 1.6)).abs() < 1e-15 && (y - (1.0 - 5.0 * x * x)).abs() < 1e-15);
        assert!((z + 0.018138).abs() < 5e-7, "{z}");
        let [x, _, z] = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..ThreeVariable::FIG5A }.equilibria(0.0).unwrap()[0];
        assert!((x - -1.6103931697431384).abs() < 5e-15 && (z - (x + 1.6)).abs() < 1e-15 && (z + 0.0104).abs() < 5e-5, "{x} {z}");
        let exact = ThreeVariable::FIG6.equilibria(0.0).unwrap();
        assert_eq!(exact.len(), 1);
        assert!(worst(&exact[0], &ThreeVariable::FIG6.rest()) < 1e-15, "{exact:?}");
    }

    /// The rest state is stable for every figure set, as p. 96 asserts: the Routh–Hurwitz conditions
    /// hold, and the three real roots of [`ThreeVariable::characteristic`] are the eigenvalues
    /// `numpy.linalg.eigvals` returns for the same Jacobian.
    ///
    /// `r = 0.001, s = 1`: −18.487503, −0.074063, −0.0017396; `r = 0.001, s = 4`: −18.487349,
    /// −0.071908, −0.0040484; `r = 0.005, s = 4`: −18.486527, −0.053643, −0.027136. The slowest is the
    /// adaptation, a time constant of 575, 247 and 37 time units. Agreement: 1.4 × 10⁻¹⁴ at worst.
    #[test]
    fn the_rest_state_is_stable_for_every_figure_set() {
        let numpy = [
            (ThreeVariable::FIG5A, [-18.487503371656846, -0.07406289828478553, -0.001739628807429652]),
            (ThreeVariable::FIG6, [-18.487349244149726, -0.0719082136889628, -0.004048440910350516]),
            (ThreeVariable::RANDOM_BURSTS, [-18.486526967267032, -0.053643273815326596, -0.02713565766669884]),
        ];
        let mut w = 0.0_f64;
        for ((f, eig), tau) in numpy.into_iter().zip([575.0, 247.0, 37.0]) {
            assert!(f.is_stable(f.x1).unwrap());
            let [c2, c1, c0] = f.characteristic(f.x1);
            let lambda = cubic_roots(c2, c1, c0);
            w = w.max(worst(&lambda, &eig));
            assert!((-1.0 / lambda[2] - tau).abs() < 0.5, "{lambda:?}");
        }
        assert!(w < 5e-14, "{w}");
    }

    /// Each Routh–Hurwitz condition is needed: where either of the other two fails alone, the
    /// point is unstable and [`ThreeVariable::is_stable`] says so.
    ///
    /// The paper's `x–y` system with `r = 10, s = 1` at `x = −2/3`: `c₂ = 49/3`, `c₁ = 72`,
    /// `c₀ = −10/3` — so `c₂c₁ > c₀` holds and `c₀ > 0` fails, and the cubic has a positive real
    /// root. `a = 1, b = 6, c = 1, d = 1` with `r = 0.1, s = 10` at `x = 2`: `c₂ = −10.9`,
    /// `c₁ = −8.1`, `c₀ = 0.2` — `c₀ > 0` and `c₂c₁ > c₀` hold and `c₂ > 0` fails, and the
    /// eigenvalues sum to `−c₂ > 0`.
    #[test]
    fn each_routh_hurwitz_condition_is_needed() {
        let fast = ThreeVariable { r: 10.0, s: 1.0, ..ThreeVariable::FIG5A };
        let [c2, c1, c0] = fast.characteristic(-2.0 / 3.0);
        assert!((c2 - 49.0 / 3.0).abs() < 1e-13 && (c1 - 72.0).abs() < 1e-13 && (c0 + 10.0 / 3.0).abs() < 1e-13, "{c2} {c1} {c0}");
        assert!(c2 * c1 > c0 && cubic_roots(c2, c1, c0).iter().any(|&l| l > 0.0));
        assert!(!fast.is_stable(-2.0 / 3.0).unwrap());
        let tilted = ThreeVariable { base: TwoVariable { a: 1.0, b: 6.0, c: 1.0, d: 1.0 }, r: 0.1, s: 10.0, x1: 0.0 };
        let [c2, c1, c0] = tilted.characteristic(2.0);
        assert!((c2 + 10.9).abs() < 1e-13 && (c1 + 8.1).abs() < 1e-13 && (c0 - 0.2).abs() < 1e-13, "{c2} {c1} {c0}");
        assert!(c0 > 0.0 && c2 * c1 > c0);
        assert!(!tilted.is_stable(2.0).unwrap());
    }

    /// The characteristic polynomial is the Jacobian's, and the Jacobian is the derivative of (15).
    ///
    /// Written through the two-variable determinant `Det(A(x))`, the coefficients are
    /// `c₂ = 1 + r − A₁₁`, `c₁ = Det(A(x)) + r(1 + s − A₁₁)` and `c₀ = r(s + Det(A(x)))` with
    /// `A₁₁ = −3a x² + 2b x`, so an e.p. of (15) can be stable only where `dI(∞)/dx > −s`. Checked at
    /// a spread of `x` for the figures' three distinct parameter sets (Figs 5c, 6 and 8 share one) and
    /// an off-paper one, `r = 0.3, s = 2.5`, where no term is small: the coefficients agree to a
    /// measured 3.6 × 10⁻¹⁵, and the Jacobian matches central differences of the field (step 10⁻⁶)
    /// to 1.2 × 10⁻¹⁰ of its largest entry.
    #[test]
    fn the_characteristic_polynomial_is_the_jacobians_and_the_jacobian_is_the_derivative() {
        let odd = ThreeVariable { base: TwoVariable { a: 0.5, b: 2.0, c: 3.0, d: 7.0 }, r: 0.3, s: 2.5, x1: -0.7 };
        let (mut poly, mut jac) = (0.0_f64, 0.0_f64);
        for f in [ThreeVariable::FIG5A, ThreeVariable::FIG5C, ThreeVariable::FIG6, ThreeVariable::FIG8, ThreeVariable::RANDOM_BURSTS, odd] {
            for x in [-2.0, -1.25, -0.5, 0.25, 1.0, 1.75] {
                let m = f.base;
                let a11 = -3.0 * m.a * x * x + 2.0 * m.b * x;
                let want = [1.0 + f.r - a11, m.determinant(x) + f.r * (1.0 + f.s - a11), f.r * (f.s + m.determinant(x))];
                poly = poly.max(worst(&f.characteristic(x), &want));
                let u = [x, -3.0, 0.4];
                let j = f.jacobian(x);
                let scale = j.iter().flatten().fold(0.0_f64, |s, v| s.max(v.abs()));
                for col in 0..3 {
                    let e = 1e-6;
                    let (mut up, mut dn) = (u, u);
                    up[col] += e;
                    dn[col] -= e;
                    let (fu, fd) = (f.field(up, 0.7), f.field(dn, 0.7));
                    for row in 0..3 {
                        jac = jac.max(((fu[row] - fd[row]) / (2.0 * e) - j[row][col]).abs() / scale);
                    }
                }
            }
        }
        assert!(poly < 1e-14 && jac < 5e-10, "{poly} {jac}");
    }

    /// Fig. 6's three regimes against `DOP853` on the same inputs — Fig. 6a spike for spike, Fig. 6b
    /// burst for burst, Fig. 6c by its count and its intervals — as a check of the implementation.
    /// Panel c's drawing is `figure_6c_is_drawn_500_time_units_after_its_caption`'s.
    ///
    /// From rest at `(x₁, y₁, 0)` under a steady current, `r = 0.001, s = 4`:
    ///
    /// - `I = 0.4` (6a): an isolated burst of eight spikes, the last at `t = 148.02`, then nothing
    ///   over the remaining 2 850 time units; every spike time within a measured 1.4 × 10⁻⁵ of the
    ///   reference's.
    /// - `I = 2` (6b): a first burst of 69 spikes, "a long burst initially in response to the
    ///   current step", then bursts of nine every 452.84 time units; each burst's first spike within
    ///   6.1 × 10⁻⁵.
    /// - `I = 4` (6c): continuous firing, 436 spikes in 6 000 time units, the interspike interval
    ///   growing from 2.993 at onset — "the frequency declining from the onset of the step" — to a
    ///   steady 22.0656, against the reference's 22.065598 to 3.1 × 10⁻⁶. At t = 1 000, where the
    ///   caption puts its panel, the interval is still growing: the twelve spikes after it are 10.667
    ///   to 12.480 apart, 11.534 on average (`DOP853`: 11.5340469).
    ///
    /// And the one e.p. of (15) under each current is stable at 0.4 and unstable at 2 and 4 — the
    /// Routh–Hurwitz verdict, with `c₂c₁ > c₀` the condition that fails — at the abscissae
    /// `numpy.roots` gives, `−1.5406198`, `−1.1488891` and `−0.4450230`, to 4.4 × 10⁻¹⁶.
    #[test]
    fn figure_6_isolated_burst_periodic_bursts_and_continuous_firing() {
        let f = ThreeVariable::FIG6;
        let [a, b, c] = ThreeVariable::FIG6_I;
        let run6a = f.simulate(f.rest(), &[(steps(3000.0), a)], H, 1.0).unwrap();
        let d6a = worst(&run6a.spikes, &REF_6A);
        let run6b = f.simulate(f.rest(), &[(steps(3000.0), b)], H, 1.0).unwrap();
        assert_eq!(burst_sizes(&run6b.spikes, 50.0).unwrap(), vec![69, 9, 9, 9, 9, 9]);
        let starts = burst_starts(&run6b.spikes, 50.0);
        let d6b = worst(&starts, &REF_6B);
        let period = (starts[5] - starts[1]) / 4.0;
        assert!((period - 452.84).abs() < 5e-3, "{period}");
        let run6c = f.simulate(f.rest(), &[(steps(6000.0), c)], H, 1.0).unwrap();
        let n = run6c.spikes.len();
        let isi = run6c.spikes[n - 1] - run6c.spikes[n - 2];
        let d6c = (isi - 22.065598082163888).abs();
        let onset = run6c.spikes[1] - run6c.spikes[0];
        assert_eq!(n, 436);
        assert!((onset - 2.993379529168225).abs() < 2e-5, "{onset}");
        let late: Vec<f64> = run6c.spikes.iter().copied().filter(|&t| t >= 1000.0).take(12).collect();
        let gaps: Vec<f64> = late.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = (late[11] - late[0]) / 11.0;
        assert!((gaps[0] - 10.667).abs() < 5e-4 && (gaps[10] - 12.480).abs() < 5e-4, "{gaps:?}");
        assert!((mean - 11.534046874183984).abs() < 3e-6, "{mean}");
        assert!(d6a < 4e-5 && d6b < 2e-4 && d6c < 1e-5, "{d6a} {d6b} {d6c}");
        let numpy = [-1.5406197925864253, -1.1488891364646183, -0.4450230378907856];
        for (i, (x, stable)) in ThreeVariable::FIG6_I.into_iter().zip(numpy.into_iter().zip([true, false, false])) {
            let eq = f.equilibria(i).unwrap();
            assert_eq!(eq.len(), 1);
            assert!((eq[0][0] - x).abs() < 2e-15, "{eq:?}");
            assert_eq!(f.is_stable(eq[0][0]).unwrap(), stable, "I = {i}");
            let [c2, c1, c0] = f.characteristic(eq[0][0]);
            assert!(c2 > 0.0 && c0 > 0.0, "only the third condition decides: {c2} {c1} {c0}");
        }
    }

    /// Near `q = 0` the trigonometric formula's argument rounds past −1, which is why it is clamped,
    /// and the two small roots are not resolved.
    ///
    /// `a = 1, b = 3, c = 2⁻⁶⁰, d = 3.875` at `I = 0` gives `p = 7/8`, `q = 2⁻⁶⁰` and a positive
    /// discriminant: three e.p.s, `−7/8` and `±√(q/p) = ±1.0 × 10⁻⁹` to first order. The argument
    /// `3Q/(P·2√(−P/3))`, computed as the solver computes it, is `−1 − 2⁻⁵¹`; unclamped, `acos` would
    /// return NaN. Clamped, the solver returns three roots — the count is right — with the pair both
    /// at `0`.
    #[test]
    fn the_cubic_formula_survives_an_argument_rounded_past_one() {
        let m = TwoVariable { a: 1.0, b: 3.0, c: 2.0_f64.powi(-60), d: 3.875 };
        assert_eq!((m.p(), m.q(0.0)), (0.875, 2.0_f64.powi(-60)));
        assert!(m.discriminant(0.0) > 0.0);
        let (big_p, big_q) = (-(0.875_f64 * 0.875) / 3.0, 2.0 * 0.875 * 0.875 * 0.875 / 27.0 - 2.0_f64.powi(-60));
        let argument = 3.0 * big_q / (big_p * (2.0 * (-big_p / 3.0).sqrt()));
        assert_eq!(argument, -1.0 - 2.0_f64.powi(-51));
        assert_eq!(m.equilibria(0.0).unwrap(), vec![-0.875, 0.0, 0.0]);
        assert!(((m.q(0.0) / m.p()).sqrt() - 1.0e-9).abs() < 5e-12);
    }

    /// Two roots `2⁻²⁶` apart with a linear term: the count is lost, and the pair comes back once at
    /// its midpoint — where the Newton step's guard is what keeps it.
    ///
    /// `(x + 1)(x + 1 − 2⁻²⁶)(x − 15/8)` expanded in `f64` has coefficients that are EXACT — checked
    /// here in integers, the roots scaled by `2²⁶` — so its roots are exactly `−1`, `−1 + 2⁻²⁶` and
    /// `15/8`, three distinct ones. The solver's discriminant is positive in exact arithmetic but
    /// below the rounding of its terms, and it returns the pair as one root within a measured
    /// 1.1 × 10⁻¹⁶ of the midpoint `−1 + 2⁻²⁷`, beside `15/8` exactly. There the slope of the cubic
    /// is nearly zero, and an unguarded Newton step would throw the root far away.
    #[test]
    fn a_pair_closer_than_the_formula_resolves_comes_back_at_its_midpoint() {
        let (r1, r2, r3) = (-1.0, -1.0 + 2.0_f64.powi(-26), 1.875);
        let (b, c, d) = (-(r1 + r2 + r3), r1 * r2 + r1 * r3 + r2 * r3, -r1 * r2 * r3);
        let (k1, k2, k3) = (-(1_i128 << 26), -(1_i128 << 26) + 1, 15_i128 << 23);
        let exact = |v: f64, shift: i32| -> i128 { (v * 2.0_f64.powi(shift)) as i128 };
        assert_eq!(exact(b, 26), -(k1 + k2 + k3));
        assert_eq!(exact(c, 52), k1 * k2 + k1 * k3 + k2 * k3);
        assert_eq!(exact(d, 78), -k1 * k2 * k3);
        let roots = cubic_roots(b, c, d);
        assert_eq!(roots.len(), 2, "{roots:?}");
        assert!((roots[0] - (-1.0 + 2.0_f64.powi(-27))).abs() < 5e-16 && roots[1] == r3, "{roots:?}");
    }

    /// The solver's other branches, each where it is exact.
    ///
    /// A triple root: `(x − 1)³` gives `Δ = 0`, `P = 0`, and `1` once — where the Newton step is
    /// `0/0` and is refused. Near a triple root `Δ` and `P` can disagree:
    /// `x³ + (3/8)x² + (3/64 + 2⁻⁵⁶)x + 1/512 + 2⁻⁵³` has `P = 2⁻⁵⁶` exactly, positive, so one real
    /// root, yet its computed `Δ` is positive, and without the guard on `P` the trigonometric
    /// formula would take the square root of `−P/3`; the solver returns `−1/8`, the three as one.
    /// The coefficients of `(x + 1/24)³` rounded to `f64` give `P = 0` exactly and a positive `Δ`,
    /// and again one root, `−1/24` as `−b/3` evaluates it (the mutation list's two guards on `P` are
    /// what these two cases catch). A zero constant term with a linear term: `x(x² − 4)` gives
    /// `−2, 0, 2`, and through [`ThreeVariable::equilibria`], where `c + I + s x₁ = 0` puts an e.p.
    /// at the origin, with `x₁ = −3/2` and `a = 1, b = 3, c = 1`: `d = 8, s = 4, I = 5` gives
    /// `x(x² + 5x + 4)`, three e.p.s at `−4, −1, 0`; `d = 5, s = 1, I = 1/2` gives `x(x + 1)²`, two;
    /// and `d = 5, s = 4, I = 5` gives `x(x² + 2x + 4)`, the origin alone. Each with `y = c − d x²`
    /// and `z = s(x − x₁)`.
    #[test]
    fn every_branch_of_the_cubic_solver_where_it_is_exact() {
        assert_eq!(cubic_roots(-3.0, 3.0, -1.0), vec![1.0]);
        let (c, d) = (3.0 / 64.0 + 2.0_f64.powi(-56), 1.0 / 512.0 + 2.0_f64.powi(-53));
        assert_eq!(c - 0.375 * 0.375 / 3.0, 2.0_f64.powi(-56));
        assert_eq!(cubic_roots(0.375, c, d), vec![-0.125]);
        let b = 0.125;
        assert_eq!(cubic_roots(b, b * b / 3.0, b * b * b / 27.0), vec![-b / 3.0]);
        assert_eq!(cubic_roots(0.0, -4.0, 0.0), vec![-2.0, 0.0, 2.0]);
        let at = |d: f64, s: f64, i: f64| {
            ThreeVariable { base: TwoVariable { a: 1.0, b: 3.0, c: 1.0, d }, r: 0.001, s, x1: -1.5 }.equilibria(i).unwrap()
        };
        assert_eq!(at(8.0, 4.0, 5.0), vec![[-4.0, -127.0, -10.0], [-1.0, -7.0, 2.0], [0.0, 1.0, 6.0]]);
        assert_eq!(at(5.0, 1.0, 0.5), vec![[-1.0, -4.0, 0.5], [0.0, 1.0, 1.5]]);
        assert_eq!(at(5.0, 4.0, 5.0), vec![[0.0, 1.0, 6.0]]);
    }

    /// The Newton step is kept when it does not raise the residual, and that includes a tie.
    ///
    /// `a = 1, b = 3, c = 1, d = 9.625` under `I = 11.1875` gives (9) as `x³ + 6.625x² = 12.1875`, a
    /// root of which the trigonometric formula puts at `−1.5496108177645913` and the Newton step at
    /// `−1.549610817764591`, whose residuals have the same size, found by a search over
    /// binary-fraction cubics. The step is the one kept.
    #[test]
    fn a_newton_step_that_ties_the_residual_is_kept() {
        let m = TwoVariable { a: 1.0, b: 3.0, c: 1.0, d: 9.625 };
        assert_eq!((m.p(), m.q(11.1875)), (6.625, 12.1875));
        let roots = m.equilibria(11.1875).unwrap();
        // Horner's form, as the solver evaluates it; `c = 0` adds nothing.
        let f = |x: f64| (x + 6.625) * x * x - 12.1875;
        let (formula, step) = (-1.5496108177645913, -1.549610817764591);
        assert_eq!(f(formula).abs(), f(step).abs(), "a tie");
        assert!(roots.contains(&step) && !roots.contains(&formula), "{roots:?}");
    }

    /// Every refusal, rendered: each names the quantity it refused and the value it was sent.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let msg = |e: HindmarshRoseError| e.to_string();
        assert_eq!(TwoVariable::new(1.0, 3.0, 1.0, 5.0).unwrap(), TwoVariable::PAPER);
        assert_eq!(msg(TwoVariable::new(0.0, 3.0, 1.0, 5.0).unwrap_err()), "a = 0 must be finite and positive");
        assert_eq!(msg(TwoVariable::new(1.0, f64::NAN, 1.0, 5.0).unwrap_err()), "b = NaN must be finite and positive");
        assert_eq!(msg(TwoVariable::new(1.0, 3.0, -1.0, 5.0).unwrap_err()), "c = -1 must be finite and positive");
        assert_eq!(msg(TwoVariable::new(1.0, 3.0, 1.0, f64::INFINITY).unwrap_err()), "d = inf must be finite and positive");
        let m = TwoVariable::PAPER;
        let bad = TwoVariable { a: -1.0, ..m };
        let refusals = [
            bad.equilibria(0.0).unwrap_err(),
            bad.three_point_condition(0.0).unwrap_err(),
            bad.band().unwrap_err(),
            bad.l().unwrap_err(),
            bad.m().unwrap_err(),
            bad.region(0.5).unwrap_err(),
            bad.region_iv_condition(0.0).unwrap_err(),
            bad.kind(0.5).unwrap_err(),
            bad.step(0.0, 0.0, 0.0, H).unwrap_err(),
            bad.simulate([0.0, 0.0], &[], H, 1.0).unwrap_err(),
            ThreeVariable::new(bad, 0.001, 4.0).unwrap_err(),
            ThreeVariable { base: bad, ..ThreeVariable::FIG6 }.simulate([0.0; 3], &[], H, 1.0).unwrap_err(),
        ];
        for e in refusals {
            assert_eq!(msg(e), "a = -1 must be finite and positive");
        }
        assert_eq!(msg(ThreeVariable::new(m, 0.0, 4.0).unwrap_err()), "r = 0 must be finite and positive");
        assert_eq!(msg(ThreeVariable::new(m, 0.001, -4.0).unwrap_err()), "s = -4 must be finite and positive");
        let f = ThreeVariable::FIG6;
        for (g, want) in [
            (ThreeVariable { r: 0.0, ..f }, "r = 0 must be finite and positive"),
            (ThreeVariable { s: f64::NAN, ..f }, "s = NaN must be finite and positive"),
            (ThreeVariable { x1: f64::NAN, ..f }, "x1 = NaN is not finite"),
        ] {
            assert_eq!(msg(g.equilibria(0.0).unwrap_err()), want);
            assert_eq!(msg(g.is_stable(-1.5).unwrap_err()), want);
            assert_eq!(msg(g.step([0.0; 3], 0.0, H).unwrap_err()), want);
            assert_eq!(msg(g.simulate([0.0; 3], &[], H, 1.0).unwrap_err()), want);
        }
        // Non-finite inputs, each by the name the call uses for it.
        assert_eq!(msg(m.equilibria(f64::NAN).unwrap_err()), "I = NaN is not finite");
        assert_eq!(msg(m.three_point_condition(f64::INFINITY).unwrap_err()), "I = inf is not finite");
        assert_eq!(msg(m.region_iv_condition(f64::NEG_INFINITY).unwrap_err()), "I = -inf is not finite");
        assert_eq!(msg(f.equilibria(f64::NAN).unwrap_err()), "I = NaN is not finite");
        assert_eq!(msg(m.region(f64::NAN).unwrap_err()), "x0 = NaN is not finite");
        assert_eq!(msg(m.kind(f64::INFINITY).unwrap_err()), "x0 = inf is not finite");
        assert_eq!(msg(f.is_stable(f64::NAN).unwrap_err()), "x = NaN is not finite");
        assert_eq!(msg(m.simulate([f64::NAN, 0.0], &[], H, 1.0).unwrap_err()), "x = NaN is not finite");
        assert_eq!(msg(m.step(0.0, f64::INFINITY, 0.0, H).unwrap_err()), "y = inf is not finite");
        assert_eq!(msg(f.simulate([0.0, 0.0, f64::NAN], &[], H, 1.0).unwrap_err()), "z = NaN is not finite");
        assert_eq!(msg(f.simulate([0.0; 3], &[(1, 0.0), (1, f64::NAN)], H, 1.0).unwrap_err()), "I = NaN is not finite");
        assert_eq!(msg(m.simulate([0.0; 2], &[], H, f64::NAN).unwrap_err()), "threshold = NaN is not finite");
        assert_eq!(msg(m.simulate([0.0; 2], &[], 0.0, 1.0).unwrap_err()), "h = 0 must be finite and positive");
        assert_eq!(msg(f.step([0.0; 3], 0.0, -0.5).unwrap_err()), "h = -0.5 must be finite and positive");
        // `D` must be real, and only strictly: `b² = 3a` exactly is refused, one ulp more is not.
        let flat = TwoVariable { a: 3.0, b: 3.0, c: 1.0, d: 5.0 };
        let want = "a = 3, b = 3 give b^2 <= 3a, and Table 1 and eqs. (10)-(11) need b^2 > 3a";
        assert_eq!(msg(flat.band().unwrap_err()), want);
        assert_eq!(msg(flat.region(0.5).unwrap_err()), want);
        assert_eq!(msg(flat.region_iv_condition(0.0).unwrap_err()), want);
        assert!(TwoVariable { b: 3.0_f64.next_up(), ..flat }.band().unwrap() > 0.0);
        assert_eq!(
            msg(TwoVariable { b: 1.0, ..flat }.band().unwrap_err()),
            "a = 3, b = 1 give b^2 <= 3a, and Table 1 and eqs. (10)-(11) need b^2 > 3a"
        );
        // Table 1 needs `b < d`, strictly, and so does condition (12), which places a root in it.
        let level = TwoVariable { a: 1.0, b: 5.0, c: 1.0, d: 5.0 };
        let want = "b = 5 is not below d = 5, and Table 1's regions are in order only when b < d";
        assert_eq!(msg(level.region(0.5).unwrap_err()), want);
        assert_eq!(msg(level.region_iv_condition(0.0).unwrap_err()), want);
        assert!(TwoVariable { d: 5.0_f64.next_up(), ..level }.region(0.5).is_ok());
        assert!(TwoVariable { d: 5.0_f64.next_up(), ..level }.region_iv_condition(0.0).is_ok());
        assert_eq!(
            msg(TwoVariable { b: 6.0, ..level }.region(0.5).unwrap_err()),
            "b = 6 is not below d = 5, and Table 1's regions are in order only when b < d"
        );
        // Before `q`: with `b = 5, d = 2`, `q = 0` at `I = −1` leaves (9) a positive root, 3.
        assert_eq!(
            msg(TwoVariable { d: 2.0, ..level }.region_iv_condition(-1.0).unwrap_err()),
            "b = 5 is not below d = 2, and Table 1's regions are in order only when b < d"
        );
        // Condition (12) needs `q > 0`: at `I = −c` exactly it is zero.
        assert_eq!(
            msg(m.region_iv_condition(-1.0).unwrap_err()),
            "q = 0 is not positive, so with b < d eq. (9) has no positive root for condition (12) to place"
        );
        assert_eq!(
            msg(m.region_iv_condition(-3.0).unwrap_err()),
            "q = -2 is not positive, so with b < d eq. (9) has no positive root for condition (12) to place"
        );
        assert!(!m.region_iv_condition(-1.0 + 1.0 / 64.0).unwrap(), "q = 1/64 puts the root below L");
        // A step too long for the fast equations: at `h = 0.5` under Fig. 6c's current the third
        // step, the one from t = 1, leaves the finite numbers.
        assert_eq!(
            msg(f.simulate(f.rest(), &[(1000, 4.0)], 0.5, 1.0).unwrap_err()),
            "the state left the finite numbers at t = 1: the step h = 0.5 is too long"
        );
        // A spike train must be finite and must not run backwards; equal times are allowed.
        assert_eq!(msg(burst_sizes(&[], 0.0).unwrap_err()), "gap = 0 must be finite and positive");
        assert_eq!(msg(burst_sizes(&[1.0, f64::NAN], 1.0).unwrap_err()), "spike time = NaN is not finite");
        assert_eq!(msg(burst_sizes(&[1.0, 3.0, 2.0], 1.0).unwrap_err()), "spike 2 at t = 2 comes before the spike ahead of it at t = 3");
        assert_eq!(burst_sizes(&[1.0, 1.0, 2.0], 0.5).unwrap(), vec![2, 1]);
    }

    /// A burst ends where two spikes are MORE than `gap` apart, and nowhere else.
    ///
    /// On binary fractions, so each boundary is exact: an interval equal to the gap stays inside
    /// the burst, and the next double below the gap splits it.
    #[test]
    fn a_burst_ends_where_the_interval_exceeds_the_gap() {
        assert_eq!(burst_sizes(&[], 1.0).unwrap(), Vec::<usize>::new());
        assert_eq!(burst_sizes(&[3.0], 1.0).unwrap(), vec![1]);
        let t = [0.0, 0.5, 1.0, 4.0, 4.25, 9.0];
        assert_eq!(burst_sizes(&t, 0.5).unwrap(), vec![3, 2, 1]);
        assert_eq!(burst_sizes(&t, 0.5_f64.next_down()).unwrap(), vec![1, 1, 1, 2, 1]);
        assert_eq!(burst_sizes(&t, 3.0).unwrap(), vec![5, 1]);
        assert_eq!(burst_sizes(&t, 4.75).unwrap(), vec![6]);
        assert_eq!(burst_sizes(&t, 4.75_f64.next_down()).unwrap(), vec![5, 1]);
    }

    /// A spike is an upward crossing, timed by linear interpolation inside its step.
    ///
    /// Two exact cases. At the rest state the field is exactly zero, so `x` sits ON a threshold of
    /// `x₁` for ever and never crosses it: no spikes. And with the threshold set to the `x` the
    /// first step lands on, the crossing is at that step's end, `t = h` exactly.
    #[test]
    fn a_spike_is_an_upward_crossing_timed_inside_its_step() {
        let f = ThreeVariable::FIG6;
        assert!(f.simulate(f.rest(), &[(100, 0.0)], H, f.x1).unwrap().spikes.is_empty());
        let start = [-1.0, -4.0, 0.0];
        let first = f.step(start, 2.0, 0.25).unwrap();
        assert!(first[0] > start[0], "{first:?}");
        let run = f.simulate(start, &[(8, 2.0)], 0.25, first[0]).unwrap();
        assert_eq!(run.spikes, vec![0.25]);
    }

    /// `step` is one step of `simulate`, bit for bit, in both models and under a current.
    #[test]
    fn a_step_is_one_step_of_a_run() {
        let m = TwoVariable::PAPER;
        let (mut x, mut y) = (0.5, -1.0);
        for _ in 0..100 {
            (x, y) = m.step(x, y, 0.75, 0.03125).unwrap();
        }
        assert_eq!(m.simulate([0.5, -1.0], &[(100, 0.75)], 0.03125, 1.0).unwrap().end, [x, y]);
        let f = ThreeVariable::FIG5A;
        let mut u = [0.5, -1.0, 0.25];
        for _ in 0..100 {
            u = f.step(u, 0.75, 0.03125).unwrap();
        }
        assert_eq!(f.simulate([0.5, -1.0, 0.25], &[(100, 0.75)], 0.03125, 1.0).unwrap().end, u);
        assert_ne!(u, f.simulate([0.5, -1.0, 0.25], &[(100, 0.0)], 0.03125, 1.0).unwrap().end, "the current reaches the step");
    }

    /// The integrator is fourth order: halving the step divides the error by sixteen.
    ///
    /// Each error is against the same run at `h = 1/8192`: 16 time units of (13)–(14) at `I = 0`
    /// from beside C, round the limit cycle and through a spike, and 4 time units of (15) from rest
    /// under Fig. 6b's `I = 2`. From `h = 1/128` to `1/256` to `1/512` the measured orders
    /// `log₂(e(h)/e(h/2))` are 4.021 and 4.011 for the two-variable model and 4.070 and 4.042 for the
    /// three.
    #[test]
    fn the_integrator_is_fourth_order() {
        let m = TwoVariable::PAPER;
        let xc = (5.0_f64.sqrt() - 1.0) / 2.0 + 1e-3;
        let two = |k: u32| m.simulate([xc, m.c - m.d * xc * xc], &[(16 << k, 0.0)], 1.0 / f64::from(1 << k), 1.0).unwrap().end;
        let f = ThreeVariable::FIG6;
        let three = |k: u32| f.simulate(f.rest(), &[(4 << k, 2.0)], 1.0 / f64::from(1 << k), 1.0).unwrap().end;
        let (r2, r3) = (two(13), three(13));
        let e2: Vec<f64> = (7..10).map(|k| worst(&two(k), &r2)).collect();
        let e3: Vec<f64> = (7..10).map(|k| worst(&three(k), &r3)).collect();
        let order = |e: &[f64]| -> Vec<f64> { e.windows(2).map(|w| (w[0] / w[1]).log2()).collect() };
        for p in order(&e2).into_iter().chain(order(&e3)) {
            assert!((p - 4.0).abs() < 0.15, "{p}");
        }
    }

    /// Fig. 3a and p. 95: a pulse of `I = 1` too short to fire the cell switches it from rest onto
    /// the limit cycle.
    ///
    /// From A under (13)–(14), each pulse taken in 1 000 RK4 steps and the 300 time units after it
    /// at `h = 0.01`: 9.8931003 is the shortest pulse that leaves the phase point above the saddle
    /// separatrix, so that it fires for ever after; anything shorter lets it slide back to A. Found
    /// by bisection; `DOP853` puts the same edge 5 × 10⁻⁹ away. Pulses from there to 13.80 switch
    /// the cell without firing it — "a shorter current pulse which does not fire the cell" (p. 93);
    /// from 13.81 the pulse fires once itself and still switches; and from 14.03 to 16.09 it fires
    /// once and the cell returns to rest, longer pulses alternating between the two. That is every
    /// pulse from 9.80 to 40.00 in hundredths, whose pair (spikes during the pulse, firing after it)
    /// changes only at 9.90, 13.81, 14.03, 16.10, 20.01, 20.24, 22.31, 26.22, 26.44, 28.51, 32.42,
    /// 32.65, 34.72, 38.62 and 38.85: the pulse fires once more at each of 13.81, 20.01, 26.22,
    /// 32.42 and 38.62, and the cell switches except from 14.03, 20.24, 26.44, 32.65 and 38.85 to
    /// just before 16.10, 22.31, 28.51, 34.72 and past 40, where it returns to rest. `DOP853` run on
    /// all 3 021 widths changes at the same fifteen. The limit cycle's period, the mean of
    /// its last ten intervals over 400 time units from beside C, is 18.634794, 1.7 × 10⁻⁶ from
    /// `DOP853`'s; after a pulse of 10 the model settles onto the same period, and the shortest
    /// switching pulse is 0.531 of it. The largest `x` the cycle reaches at the steps of the run is
    /// 1.686016, and Fig. 3a's drawn spikes peak near 1.7.
    #[test]
    fn figure_3a_a_pulse_that_does_not_fire_switches_rest_onto_the_limit_cycle() {
        let m = TwoVariable::PAPER;
        let a = [ThreeVariable::X1, m.c - m.d * ThreeVariable::X1 * ThreeVariable::X1];
        let fires = |w: f64| -> (usize, Vec<f64>) {
            let pulse = m.simulate(a, &[(1000, TwoVariable::PULSE_I)], w / 1000.0, 1.0).unwrap();
            (pulse.spikes.len(), m.simulate(pulse.end, &[(steps(300.0), 0.0)], H, 1.0).unwrap().spikes)
        };
        let (mut lo, mut hi) = (9.0, 10.0);
        assert!(fires(lo).1.is_empty() && !fires(hi).1.is_empty());
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if fires(mid).1.is_empty() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (during, _) = fires(hi);
        let (during10, after) = fires(10.0);
        let n = after.len();
        let settled = (after[n - 1] - after[n - 11]) / 10.0;
        let xc = (5.0_f64.sqrt() - 1.0) / 2.0 + 1e-3;
        let cycle = m.simulate([xc, m.c - m.d * xc * xc], &[(steps(400.0), 0.0)], H, 1.0).unwrap().spikes;
        let k = cycle.len();
        let period = (cycle[k - 1] - cycle[k - 11]) / 10.0;
        let reference = 18.634795513605134;
        assert_eq!((during, during10), (0, 0));
        assert!((hi - 9.89310031450077).abs() < 3e-8, "{hi}");
        assert!((period - reference).abs() < 5e-6 && (settled - reference).abs() < 5e-6, "{period} {settled}");
        assert!((hi / period - 0.531).abs() < 5e-4, "{}", hi / period);
        assert!(m.simulate(a, &[(steps(300.0), 0.0)], H, 1.0).unwrap().spikes.is_empty(), "A is at rest");
        // The whole grid, each run after the pulse stopped at its first spike: 300 time units in
        // twelve pieces of 25, which take the same steps as one run of 300.
        let switches = |k: u32| -> (u32, usize, bool) {
            let pulse = m.simulate(a, &[(1000, TwoVariable::PULSE_I)], f64::from(k) / 100.0 / 1000.0, 1.0).unwrap();
            let mut u = pulse.end;
            for _ in 0..12 {
                let piece = m.simulate(u, &[(steps(25.0), 0.0)], H, 1.0).unwrap();
                if !piece.spikes.is_empty() {
                    return (k, pulse.spikes.len(), true);
                }
                u = piece.end;
            }
            (k, pulse.spikes.len(), false)
        };
        let mut bands: Vec<(u32, usize, bool)> = Vec::new();
        for (k, during, after) in (980..=4000).map(switches) {
            if bands.last().is_none_or(|&(_, d, f)| (d, f) != (during, after)) {
                bands.push((k, during, after));
            }
        }
        let want = [
            (980, 0, false),
            (990, 0, true),
            (1381, 1, true),
            (1403, 1, false),
            (1610, 1, true),
            (2001, 2, true),
            (2024, 2, false),
            (2231, 2, true),
            (2622, 3, true),
            (2644, 3, false),
            (2851, 3, true),
            (3242, 4, true),
            (3265, 4, false),
            (3472, 4, true),
            (3862, 5, true),
            (3885, 5, false),
        ];
        assert_eq!(bands, want);
        assert_eq!(switches(1380), (1380, fires(13.80).0, !fires(13.80).1.is_empty()), "the pieces are one run");
        let (mut u, mut top) = ([xc, m.c - m.d * xc * xc], f64::NEG_INFINITY);
        for k in 0..steps(400.0) {
            u = m.step(u[0], u[1], 0.0, H).map(|(x, y)| [x, y]).unwrap();
            if k >= steps(300.0) {
                top = top.max(u[0]);
            }
        }
        assert!((top - 1.686016).abs() < 5e-7, "{top}");
    }

    /// Fig. 5: a short pulse of `I = 1` from rest gives at most five spikes with `s = 1`, and one with
    /// `s = 4` over the pulses the figure draws, and afterwards `x` dips below `x₁` and returns to it
    /// (p. 96).
    ///
    /// `r = 0.001, s = 1` (5a), pulses of 8 to 24 time units in quarters, counting every spike in
    /// the pulse and the 1 500 time units after it: none below 10, then 2, 4, 5, 4, 1, 4, 5, 2 and 5 in
    /// bands, never more than five, and four for the drawn pulse of 11–12; `DOP853` gives the same
    /// count at every width. The figure draws eight, all after the pulse. Going on in quarters, the
    /// first width that gives eight is 44.75, with five of them inside the pulse, and 44.5 gives seven
    /// — `DOP853` the same at both. `r = 0.001, s = 4` (5c), pulses of 10 to 15 in quarters: none to
    /// 10.25, one from 10.5 to 13, two from 13.25 to 13.75 and one from 14. The figure draws one, after
    /// a pulse of 10.99 or 11.56 by the two readings of its "20" bar (edges 94.5 px apart between the
    /// centres of its two strokes, against the bar's 163.5 px between tick centres and 172 px end to
    /// end, on the p. 96 scan). With `x₁ = −1.6` from `(−1.6, −11.8, 0)`: one from 10 to 12.5, two
    /// from 12.75 to 13.5 and one from 13.75; from that model's own rest, one to 10.75, two from 11
    /// to 13.5 and one from 13.75. `DOP853` gives the same count at every quarter from all three
    /// starts. At the two drawn widths the exact root fires once, the rounding from `(−1.6, −11.8, 0)`
    /// once, and the rounding from its own rest twice, having started to at a pulse of 10.915601 —
    /// found by bisection, the pulse taken in 1 000 steps, a measured 6.2 × 10⁻⁷ from `DOP853`'s
    /// 10.9156015.
    ///
    /// After a pulse of 12, 5c's `x` plateaus near −0.93 — the depolarizing afterpotential — then
    /// falls to −1.7034971 at t = 113.5, below `x₁`; 5a's falls to −1.6811483 at t = 205.4. The dips
    /// are within a measured 3.6 × 10⁻⁹ and 7.8 × 10⁻⁹ of `DOP853`'s. At t = 6 012 both are back near
    /// `x₁`, 5a still 3.2 × 10⁻⁶ below it — "the x value returns very slowly to the original value
    /// x₁" — and each within 2.4 × 10⁻¹³ of `DOP853`'s.
    #[test]
    fn figure_5_a_short_pulse_gives_a_short_burst_or_a_single_spike() {
        let count = |f: &ThreeVariable, w: f64| {
            f.simulate(f.rest(), &[(steps(w), TwoVariable::PULSE_I), (steps(1500.0), 0.0)], H, 1.0).unwrap().spikes.len()
        };
        let fig5a: Vec<usize> = (32..=96).map(|q| count(&ThreeVariable::FIG5A, f64::from(q) / 4.0)).collect();
        let mut want = vec![0; 8];
        for (n, c) in [(1, 2), (12, 4), (3, 5), (1, 4), (9, 1), (2, 4), (13, 5), (10, 2), (6, 5)] {
            want.extend(core::iter::repeat_n(c, n));
        }
        assert_eq!(fig5a, want);
        assert_eq!(fig5a[12..=16], [4; 5], "the drawn pulse, 11 to 12");
        let most = (97..=178).map(|q| count(&ThreeVariable::FIG5A, f64::from(q) / 4.0)).max();
        let f = ThreeVariable::FIG5A;
        let eight = f.simulate(f.rest(), &[(steps(44.75), TwoVariable::PULSE_I), (steps(1500.0), 0.0)], H, 1.0).unwrap().spikes;
        assert_eq!((most, eight.len(), eight.iter().filter(|&&t| t < 44.75).count()), (Some(7), 8, 5));
        assert_eq!(count(&ThreeVariable::FIG5A, 44.5), 7);
        let printed = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..ThreeVariable::FIG5C };
        let own = printed.equilibria(0.0).unwrap()[0];
        let quarters = |f: &ThreeVariable, start: [f64; 3]| -> Vec<usize> {
            (40..=60)
                .map(|q| {
                    let w = f64::from(q) / 4.0;
                    f.simulate(start, &[(steps(w), TwoVariable::PULSE_I), (steps(1500.0), 0.0)], H, 1.0).unwrap().spikes.len()
                })
                .collect()
        };
        let bands = |runs: &[(usize, usize)]| -> Vec<usize> { runs.iter().flat_map(|&(n, c)| core::iter::repeat_n(c, n)).collect() };
        assert_eq!(quarters(&ThreeVariable::FIG5C, ThreeVariable::FIG5C.rest()), bands(&[(2, 0), (11, 1), (3, 2), (5, 1)]));
        assert_eq!(quarters(&printed, printed.rest()), bands(&[(11, 1), (4, 2), (6, 1)]));
        assert_eq!(quarters(&printed, own), bands(&[(4, 1), (11, 2), (6, 1)]));
        // The drawn pulse by the bar's two readings, 10.99 and 11.56, each a whole number of steps.
        let drawn = |f: &ThreeVariable, start: [f64; 3]| -> [usize; 2] {
            [10.99, 11.56].map(|w| f.simulate(start, &[(steps(w), TwoVariable::PULSE_I), (steps(1500.0), 0.0)], H, 1.0).unwrap().spikes.len())
        };
        let fig5c = ThreeVariable::FIG5C;
        assert_eq!((drawn(&fig5c, fig5c.rest()), drawn(&printed, printed.rest()), drawn(&printed, own)), ([1, 1], [1, 1], [2, 2]));
        // Where the rounding from its own rest starts to fire twice: the pulse in 1 000 steps.
        let pulse = |w: f64| -> usize {
            let during = printed.simulate(own, &[(1000, TwoVariable::PULSE_I)], w / 1000.0, 1.0).unwrap();
            during.spikes.len() + printed.simulate(during.end, &[(steps(1500.0), 0.0)], H, 1.0).unwrap().spikes.len()
        };
        let (mut lo, mut hi) = (10.75, 11.0);
        assert_eq!((pulse(lo), pulse(hi)), (1, 2));
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if pulse(mid) < 2 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!((hi - 10.915601490298286).abs() < 3e-6, "{hi}");
        let after = |f: &ThreeVariable, from: f64| -> (f64, f64, f64, f64) {
            let mut u = f.rest();
            for _ in 0..steps(12.0) {
                u = f.step(u, TwoVariable::PULSE_I, H).unwrap();
            }
            let (mut low, mut when, mut plateau) = (f64::INFINITY, 0.0, f64::NEG_INFINITY);
            for k in 1..=steps(6000.0) {
                u = f.step(u, 0.0, H).unwrap();
                let t = 12.0 + k as f64 * H;
                if t > from && u[0] < low {
                    (low, when) = (u[0], t);
                }
                if (30.0..40.0).contains(&t) {
                    plateau = plateau.max(u[0]);
                }
            }
            (low, when, u[0], plateau)
        };
        let (low_a, when_a, end_a, _) = after(&ThreeVariable::FIG5A, 150.0);
        let (low_c, when_c, end_c, plateau) = after(&ThreeVariable::FIG5C, 30.0);
        assert!((low_a - -1.6811483104080502).abs() < 3e-8 && (when_a - 205.4).abs() < 0.01, "{low_a} {when_a}");
        assert!((low_c - -1.7034971209305487).abs() < 3e-8 && (when_c - 113.52).abs() < 0.01, "{low_c} {when_c}");
        assert!(low_a < ThreeVariable::X1 && low_c < ThreeVariable::X1 && (plateau - -0.93).abs() < 5e-3, "{plateau}");
        assert!((end_a - -1.6180371818418275).abs() < 1e-12 && (end_c - -1.618033988755549).abs() < 1e-12, "{end_a} {end_c}");
        assert!((end_a - ThreeVariable::X1 + 3.2e-6).abs() < 1e-7);
    }

    /// ⚠ What the rounding `x₁ = −1.6` changes in Fig. 6, each run against `DOP853`'s with the same
    /// rounding.
    ///
    /// Started as a reader using the printed value would, at `(−1.6, −11.8, 0)`, which
    /// [`ThreeVariable::rest`] gives for that `x₁`: Fig. 6a still gives eight spikes, but the ratio
    /// of its last to its first interspike interval falls from 2.504 to 2.244; Fig. 6b's first burst
    /// gains a spike, 69 to 70, the burst period falls from 452.84 to 430.78, and a periodic burst's
    /// duration over the period from 0.3113 to 0.3025 (the exact root's burst lasting 140.98 time
    /// units, the step p. 99 gives Fig. 8); Fig. 6c's steady interval falls from 22.066 to 20.735, and
    /// its 6 000 time units hold 453 spikes instead of 436. Two more ratios: the first burst start
    /// after the caption's 700 time units, over the period, which the drawing can be measured by,
    /// falls from 0.3369 to 0.3190; and Fig. 6c's steady interval over Fig. 6b's period, which the
    /// drawing does not show (`figure_6c_is_drawn_500_time_units_after_its_caption`), from 0.0487 to
    /// 0.0481. Every spike time is within a measured 1.9 × 10⁻⁵ of `DOP853`'s.
    #[test]
    fn the_printed_rounding_of_x1_moves_figure_6() {
        let printed = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..ThreeVariable::FIG6 };
        assert_eq!(printed.rest(), [-1.6, 1.0 - 5.0 * 2.56, 0.0]);
        let [a, b, c] = ThreeVariable::FIG6_I;
        let ratio = |s: &[f64]| (s[7] - s[6]) / (s[1] - s[0]);
        let f = ThreeVariable::FIG6;
        let exact6a = f.simulate(f.rest(), &[(steps(3000.0), a)], H, 1.0).unwrap().spikes;
        let run6a = printed.simulate(printed.rest(), &[(steps(3000.0), a)], H, 1.0).unwrap().spikes;
        let d6a = worst(&run6a, &REF_6A_PRINTED);
        // A burst's duration is its first spike to its last; its period, first spike to first spike.
        let shape = |s: &[f64]| -> (Vec<usize>, f64, f64, f64, f64) {
            let sizes = burst_sizes(s, 50.0).unwrap();
            let starts = burst_starts(s, 50.0);
            let lasts = s[sizes[0] + sizes[1] - 1] - s[sizes[0]];
            let period = (starts[5] - starts[1]) / 4.0;
            (sizes, lasts / (starts[2] - starts[1]), period, lasts, (starts[1] - 700.0) / period)
        };
        let exact6b = f.simulate(f.rest(), &[(steps(3000.0), b)], H, 1.0).unwrap().spikes;
        let run6b = printed.simulate(printed.rest(), &[(steps(3000.0), b)], H, 1.0).unwrap().spikes;
        let (sizes, duty, period, _, offset) = shape(&run6b);
        let (_, exact_duty, exact_period, lasts, exact_offset) = shape(&exact6b);
        let d6b = worst(&burst_starts(&run6b, 50.0), &REF_6B_PRINTED);
        let run6c = printed.simulate(printed.rest(), &[(steps(6000.0), c)], H, 1.0).unwrap().spikes;
        let n = run6c.len();
        let d6c = (run6c[n - 1] - run6c[n - 2] - 20.734955540069677).abs();
        let exact6c = f.simulate(f.rest(), &[(steps(6000.0), c)], H, 1.0).unwrap().spikes;
        let steady = |s: &[f64]| s[s.len() - 1] - s[s.len() - 2];
        assert_eq!((run6a.len(), sizes, n), (8, vec![70, 9, 9, 9, 9, 9, 1], 453));
        assert!((ratio(&run6a) - 2.244).abs() < 1e-3 && (ratio(&exact6a) - 2.504).abs() < 1e-3);
        assert!((period - 430.775).abs() < 5e-3 && (exact_period - 452.842).abs() < 5e-3, "{period} {exact_period}");
        assert!((duty - 0.3025).abs() < 5e-4 && (exact_duty - 0.3113).abs() < 5e-4, "{duty} {exact_duty}");
        assert!((lasts - 140.98).abs() < 5e-3, "{lasts}");
        assert!((offset - 0.3190).abs() < 5e-5 && (exact_offset - 0.3369).abs() < 5e-5, "{offset} {exact_offset}");
        let (ratio6c, exact_ratio6c) = (steady(&run6c) / period, steady(&exact6c) / exact_period);
        assert!((ratio6c - 0.0481).abs() < 5e-5 && (exact_ratio6c - 0.0487).abs() < 5e-5, "{ratio6c} {exact_ratio6c}");
        assert!(d6a < 4e-5 && d6b < 6e-5 && d6c < 2e-5, "{d6a} {d6b} {d6c}");
    }

    /// ⚠ Fig. 6c: the drawn firing is the model's some 500 time units after the caption's "after
    /// 1000 time units of continuous firing".
    ///
    /// The drawn twelve spikes average 0.0440 of Fig. 6b's period (28.2 px over 642 on the p. 98
    /// scan), and 18.7 or 19.8 time units by the two readings of the "100" bar. Under `I = 4` from
    /// rest, with Fig. 6b's period of 452.842: the twelve spikes from t = 1 000 average 0.02547 of
    /// it; twelve spikes first average 0.0440 of it from t = 1 491.720 to 1 712.631; the first
    /// interval at least that long starts at 1 569.049; and the first at least 18.7 starts at
    /// 1 491.720, the first at least 19.8 at 1 569.049. With `x₁ = −1.6` from `(−1.6, −11.8, 0)` and
    /// its period of 430.775: 0.02572 from t = 1 000; twelve first averaging 0.0440 from 1 550.873 to
    /// 1 760.685; the first interval that long starting at 1 624.962; and the first at least 18.7 and
    /// 19.8 starting at 1 606.164 and 1 760.685. Against `DOP853`'s run of the same, which picks out
    /// the same spikes: each time within a measured 3.0 × 10⁻⁵, each period within 1.4 × 10⁻⁵ and each
    /// ratio from t = 1 000 within 8.5 × 10⁻¹⁰. Nor is the drawn firing the "steady repetitive
    /// firing" of p. 97: over the model's twelve the interval grows at every spike, from 18.94 to
    /// 21.01 (18.24 to 19.76 with the rounding), and ends more than 0.9 short of the steady 22.066
    /// (20.735).
    #[test]
    fn figure_6c_is_drawn_500_time_units_after_its_caption() {
        let printed = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..ThreeVariable::FIG6 };
        let [_, b, c] = ThreeVariable::FIG6_I;
        let mut got = Vec::new();
        let steady = [22.065598082163888, 20.734955540069677];
        for ((f, start), steady) in [(ThreeVariable::FIG6, ThreeVariable::FIG6.rest()), (printed, printed.rest())].into_iter().zip(steady) {
            let starts = burst_starts(&f.simulate(start, &[(steps(3000.0), b)], H, 1.0).unwrap().spikes, 50.0);
            let period = (starts[5] - starts[1]) / 4.0;
            let s = f.simulate(start, &[(steps(2500.0), c)], H, 1.0).unwrap().spikes;
            // The spike that starts the first interval at least `v` long.
            let reach = |v: f64| s.windows(2).find(|w| w[1] - w[0] >= v).expect("reached within 2 500")[0];
            let k = (0..s.len() - 11).find(|&k| (s[k + 11] - s[k]) / 11.0 >= 0.0440 * period).expect("reached within 2 500");
            let j = s.iter().position(|&t| t >= 1000.0).unwrap();
            let drawn: Vec<f64> = s[k..=k + 11].windows(2).map(|w| w[1] - w[0]).collect();
            assert!(drawn.windows(2).all(|w| w[1] > w[0]) && drawn[10] < steady - 0.9, "not yet steady: {drawn:?}");
            got.push([period, (s[j + 11] - s[j]) / 11.0 / period, s[k], s[k + 11], reach(0.0440 * period), reach(28.2 / 151.0 * 100.0), reach(28.2 / 142.5 * 100.0)]);
        }
        let dop853 = [
            [452.8419512481652, 0.0254703585707834, 1491.7200317043735, 1712.6313268182425, 1569.0486793312518, 1491.7200317043735, 1569.0486793312518],
            [430.7754996756217, 0.025716325504572125, 1550.8733505574617, 1760.6849647575223, 1624.9616954131393, 1606.1643782678502, 1760.6849647575223],
        ];
        let (mut period, mut ratio, mut time) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (g, d) in got.iter().zip(&dop853) {
            period = period.max((g[0] - d[0]).abs());
            ratio = ratio.max((g[1] - d[1]).abs());
            time = time.max(worst(&g[2..], &d[2..]));
        }
        assert!(period < 5e-5 && ratio < 3e-9 && time < 1e-4, "{period} {ratio} {time}");
        assert!((got[0][0] - 452.842).abs() < 5e-4 && (got[1][0] - 430.775).abs() < 5e-4, "{got:?}");
        assert!((got[0][1] - 0.02547).abs() < 5e-6 && (got[1][1] - 0.02572).abs() < 5e-6, "{got:?}");
    }

    /// ⚠ Fig. 6a's tail does not recover to `x₁` as the text and the drawing do: under its steady
    /// `I = 0.4` it settles at the one e.p. of (15), 0.077 above `x₁`, after dipping below it.
    ///
    /// RK4 at `h = 0.01` from rest: after the burst `x` falls to −1.7006200 at t = 224.32, below
    /// `x₁ = −1.618034` and within 5 × 10⁻⁹ of `DOP853`'s dip; it is at −1.672 at t = 300, −1.634 at
    /// 400, −1.553 at 856 and −1.542 at 1 205, each within a measured 5.5 × 10⁻⁹ of `DOP853`'s, where
    /// the drawn tail holds between −1.68 and −1.65; and at t = 6 000 it is within 5.8 × 10⁻¹³ of the
    /// e.p. at `x = −1.5406198`, which is `x₁ + 0.0774142`. Before the burst, `x` crosses −1.46 at
    /// t = 7.900 (`DOP853`: 7.9003055, 3.2 × 10⁻⁸ away), 34.8 before the first spike, where the drawn
    /// trace, which starts there, takes 57–63.
    #[test]
    fn figure_6a_settles_where_its_current_puts_the_equilibrium_not_at_x1() {
        let f = ThreeVariable::FIG6;
        let i = ThreeVariable::FIG6_I[0];
        let eq = f.equilibria(i).unwrap()[0];
        let mut u = f.rest();
        let (mut low, mut when, mut tail) = (f64::INFINITY, 0.0, Vec::new());
        for k in 1..=steps(6000.0) {
            u = f.step(u, i, H).unwrap();
            let t = k as f64 * H;
            if t > 150.0 && u[0] < low {
                (low, when) = (u[0], t);
            }
            if [30_000, 40_000, 85_600, 120_500].contains(&k) {
                tail.push(u[0]);
            }
        }
        assert!(low < f.x1 && (low - -1.7006200087718029).abs() < 2e-8 && (when - 224.32).abs() < 0.01, "{low} {when}");
        let dop853 = [-1.6720258511902848, -1.6343382410395122, -1.552607338165268, -1.5424218053864123];
        assert!(worst(&tail, &dop853) < 2e-8, "{tail:?}");
        assert!((u[0] - eq[0]).abs() < 2e-12, "{u:?} {eq:?}");
        assert!((eq[0] - f.x1 - 0.0774142).abs() < 1e-7);
        let rising = f.simulate(f.rest(), &[(steps(100.0), i)], H, -1.46).unwrap().spikes;
        assert!((rising[0] - 7.900305480423285).abs() < 1e-7 && (REF_6A[0] - rising[0] - 34.77).abs() < 5e-3, "{rising:?}");
    }

    /// ⚠ Fig. 8: a hyperpolarizing step of `I = −3` gives a rebound burst, and the rounding of `x₁`
    /// all but gives the figure's nine spikes, but no step gives them as soon after release as drawn.
    ///
    /// Steps of whole time units from 140 to 226, then `I = 0` for 1 500. Nothing fires during any
    /// step, and the first rebound spike comes sooner after release at every longer step. With the
    /// exact root the step gives six rebound spikes to 145, seven from 146, eight from 165, nine from
    /// 184, ten from 205 and eleven from 226. The drawn step measures 159–169 — 1 091 px on the
    /// p. 100 scan, against the "20" bar's 137 px end to end and 129 px between its tick centres —
    /// which gives seven or eight, the first 56.12 after release at 159 and 52.47 at 169, where the
    /// figure draws it 262–262.5 px, 38.2–40.7, after. p. 99 holds the step "for a period similar to
    /// the burst duration", and a step of 141, the 140.98 a periodic burst of Fig. 6b lasts, gives
    /// six. With `x₁ = −1.6` from `(−1.6, −11.8, 0)`: seven to 156,
    /// eight from 157, nine from 176, ten from 195 and eleven from 215; from that model's own rest,
    /// seven to 150, eight from 151, nine from 170 — one past the drawn 169, which gives eight, the
    /// first 48.947 after release — ten from 189 and eleven from 209. The last step to give nine, 204,
    /// 194 and 188 in that order, has its first spike 43.505, 44.272 and 44.276 after release, and the
    /// first step to bring it within the drawn 40.7, 220, 213 and 207, gives ten. `DOP853` gives the
    /// same count at every step of the three sweeps, has the first spike sooner at every longer step,
    /// and puts the first spikes quoted here within a measured 1.6 × 10⁻⁵ of these.
    #[test]
    fn figure_8_nine_rebound_spikes_never_come_as_soon_as_drawn() {
        let rebound = |f: &ThreeVariable, start: [f64; 3], w: u32| -> (usize, usize, f64) {
            let w = f64::from(w);
            let spikes = f.simulate(start, &[(steps(w), ThreeVariable::FIG8_I), (steps(1500.0), 0.0)], H, 1.0).unwrap().spikes;
            (spikes.iter().filter(|&&t| t < w).count(), spikes.len(), spikes[0] - w)
        };
        let exact = ThreeVariable::FIG8;
        let printed = ThreeVariable { x1: ThreeVariable::X1_PRINTED, ..exact };
        let own = printed.equilibria(0.0).unwrap()[0];
        // `DOP853`'s first spike after release: at 159, at 169, and at the last step to give nine.
        let dop853 = [
            [56.12252770437283, 52.46598198873804, 43.50548523726897],
            [54.12114079619036, 50.71844492732458, 44.27198338328057],
            [52.01357617116153, 48.947021742673144, 44.27635546998738],
        ];
        let mut far = 0.0_f64;
        for ((f, start, first, edges, nine, soon), d) in [
            (exact, exact.rest(), 6, [146, 165, 184, 205, 226], 204, 220),
            (printed, printed.rest(), 7, [157, 176, 195, 215, 227], 194, 213),
            (printed, own, 7, [151, 170, 189, 209, 227], 188, 207),
        ]
        .into_iter()
        .zip(dop853)
        {
            let runs: Vec<(usize, usize, f64)> = (140..=226).map(|w| rebound(&f, start, w)).collect();
            let at = |w: usize| runs[w - 140];
            let counts: Vec<usize> = runs.iter().map(|r| r.1).collect();
            let want: Vec<usize> = (140..=226).map(|w| first + edges.iter().filter(|&&e| e <= w).count()).collect();
            assert_eq!((runs.iter().map(|r| r.0).sum::<usize>(), counts), (0, want));
            assert!(runs.windows(2).all(|r| r[1].2 < r[0].2), "the first spike comes sooner at every longer step");
            assert_eq!((at(nine).1, at(nine + 1).1), (9, 10));
            assert_eq!((140..=226).find(|&w| at(w).2 <= 40.7), Some(soon));
            assert_eq!(at(soon).1, 10);
            far = far.max(worst(&[at(159).2, at(169).2, at(nine).2], &d));
        }
        assert!(far < 5e-5, "{far}");
        assert_eq!(rebound(&exact, exact.rest(), 141).1, 6);
    }

    /// p. 98's random bursts: irregular burst sizes from a deterministic model, and the reason no
    /// run can be expected to reproduce the printed sequence.
    ///
    /// From rest under `I = 3.25` with `r = 0.005, s = 4`, the first 20 spike times agree with
    /// `DOP853`'s to a measured 2.6 × 10⁻⁵. Over 6 000 time units, splitting where two spikes are more
    /// than 60 apart — in this run the same split as "`x` fell below −1 between them", and inside
    /// the widest hole of its interval distribution, 57.1 to 65.9 — a first burst of 31 is followed
    /// by 36 bursts of 3 to 7 spikes: five of 3, five of 4, nineteen of 5, one of 6 and six of 7, with
    /// no period up to 12. The printed seventeen are two of 3, two of 4, eight of 5, one of 6 and four
    /// of 7. `DOP853`'s run agrees on the first eight bursts and not after; its own intervals have
    /// their widest hole from 40.1 to 49.0, and it too splits at a gap of 60 exactly as its troughs
    /// below −1 split it, which is how `REF_RANDOM_SIZES` was made.
    ///
    /// The regime amplifies differences: from the state at t = 2 000, adding `10⁻¹⁵` to `z` moves some
    /// later spike by more than one time unit after 2 581 time units, `10⁻¹³` after 1 897, `10⁻¹¹`
    /// after 1 587 and `10⁻⁹` after 1 205 — sooner for every hundredfold, a millionfold buying
    /// 1 376 time units, an average exponential rate of `ln 10⁶ / 1 376 = 0.0100` per time unit. The
    /// same equations at `h = 0.01` and `h = 0.005` part company at the 60th spike.
    #[test]
    fn the_random_burst_regime_is_irregular_and_amplifies_the_smallest_difference() {
        let f = ThreeVariable::RANDOM_BURSTS;
        let i = ThreeVariable::RANDOM_BURSTS_I;
        let run = f.simulate(f.rest(), &[(steps(6000.0), i)], H, 1.0).unwrap().spikes;
        let d20 = worst(&run[..20], &REF_RANDOM_20);
        let sizes = burst_sizes(&run, 60.0).unwrap();
        // The same split by the trough: a new burst wherever x falls below −1 between two spikes.
        let (mut u, mut low, mut by_trough, mut current) = (f.rest(), f64::INFINITY, Vec::new(), 0);
        for _ in 0..steps(6000.0) {
            let next = f.step(u, i, H).unwrap();
            if u[0] < 1.0 && next[0] >= 1.0 {
                if current > 0 && low < -1.0 {
                    by_trough.push(current);
                    current = 0;
                }
                current += 1;
                low = f64::INFINITY;
            }
            low = low.min(next[0]);
            u = next;
        }
        by_trough.push(current);
        let mut isi: Vec<f64> = run.windows(2).map(|w| w[1] - w[0]).collect();
        isi.sort_by(f64::total_cmp);
        let hole = isi.windows(2).filter(|w| w[0] > 20.0).fold((0.0, 0.0), |best: (f64, f64), w| if w[1] - w[0] > best.1 - best.0 { (w[0], w[1]) } else { best });
        let mut histogram = [0_usize; 9];
        for &s in &sizes[1..] {
            histogram[s.min(8)] += 1;
        }
        let periodic = (1..=12).any(|p| sizes[1..].windows(p + 1).all(|w| w[0] == w[p]));
        let mid = f.simulate(f.rest(), &[(steps(2000.0), i)], H, 1.0).unwrap().end;
        let base = f.simulate(mid, &[(steps(3000.0), i)], H, 1.0).unwrap().spikes;
        let parts = |e: f64| -> f64 {
            let mut v = mid;
            v[2] += e;
            let other = f.simulate(v, &[(steps(3000.0), i)], H, 1.0).unwrap().spikes;
            *base.iter().zip(&other).find(|(a, b)| (*a - *b).abs() > 1.0).expect("parts company within 3 000").0
        };
        let div: Vec<f64> = [1e-15, 1e-13, 1e-11, 1e-9].into_iter().map(parts).collect();
        let fine = f.simulate(f.rest(), &[(2 * steps(6000.0), i)], H / 2.0, 1.0).unwrap().spikes;
        let half = run.iter().zip(&fine).position(|(a, b)| (a - b).abs() > 1.0);
        assert!(d20 < 1e-4, "{d20}");
        assert_eq!(sizes, by_trough);
        assert_eq!((sizes[0], sizes.len(), histogram), (31, 37, [0, 0, 0, 5, 5, 19, 1, 6, 0]));
        assert!(!periodic && (hole.0 - 57.07).abs() < 0.01 && (hole.1 - 65.90).abs() < 0.01, "{hole:?}");
        assert_eq!(sizes[..8], REF_RANDOM_SIZES[..8]);
        assert_ne!(sizes[..REF_RANDOM_SIZES.len()], REF_RANDOM_SIZES);
        for (got, want) in div.iter().zip([2581.34, 1896.96, 1587.49, 1205.30]) {
            assert!((got - want).abs() < 0.01, "{div:?}");
        }
        assert!(((1e6_f64).ln() / (div[0] - div[3]) - 0.0100).abs() < 5e-5, "{div:?}");
        assert_eq!(half, Some(59));
    }

    /// [`ThreeVariable::new`] takes `x₁` as the leftmost root of (9), so it builds the figure
    /// constants exactly, and for another model its rest is an e.p. too.
    ///
    /// `a = 0.5, b = 2, c = 3, d = 7` has `p = 10, q = 6`, three e.p.s since `27q = 162 < 4p³`, and
    /// with `r = 0.3, s = 2.5` one e.p. of (15) at `I = 0`: the rest state, where the field
    /// measures 1.1 × 10⁻¹³ against a `y` of −688.
    #[test]
    fn new_puts_the_rest_state_at_the_leftmost_equilibrium() {
        let m = TwoVariable::PAPER;
        assert_eq!(ThreeVariable::new(m, 0.001, 1.0).unwrap(), ThreeVariable::FIG5A);
        assert_eq!(ThreeVariable::new(m, 0.001, 4.0).unwrap(), ThreeVariable::FIG6);
        assert_eq!(ThreeVariable::new(m, 0.005, 4.0).unwrap(), ThreeVariable::RANDOM_BURSTS);
        let odd = TwoVariable::new(0.5, 2.0, 3.0, 7.0).unwrap();
        let roots = odd.equilibria(0.0).unwrap();
        let f = ThreeVariable::new(odd, 0.3, 2.5).unwrap();
        assert_eq!((roots.len(), f.x1, f.r, f.s), (3, roots[0], 0.3, 2.5));
        let at = f.field(f.rest(), 0.0);
        let eq = f.equilibria(0.0).unwrap();
        assert!(at.iter().all(|v| v.abs() < 5e-13), "{at:?}");
        assert_eq!(eq.len(), 1);
        assert!(worst(&eq[0], &f.rest()) < 1e-14, "{eq:?}");
    }
}
