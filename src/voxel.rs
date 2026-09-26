//! Event tensors: the voxel grid and the time-binned event frames a network is fed from an event
//! camera, built to the definitions they are cited by and checked against the Python library that
//! builds them for the public event datasets.
//!
//! # What is here
//!
//! - **The discretized event volume** of Zhu, Yuan, Chaney and Daniilidis, *Unsupervised
//!   Event-based Learning of Optical Flow, Depth, and Egomotion*, CVPR 2019, section 3.1: "we scale
//!   the timestamps to the range `[0, B − 1]`",
//!
//!   ```text
//!   t*ᵢ = (B − 1)(tᵢ − t₁)/(t_N − t₁)                       (1)
//!   V(x, y, t) = Σᵢ pᵢ k_b(x − xᵢ) k_b(y − yᵢ) k_b(t − t*ᵢ)      (2)
//!   k_b(a) = max(0, 1 − |a|)                                (3)
//!   ```
//!
//!   with `pᵢ = ±1`. On integer pixel coordinates the spatial kernels select one pixel, and the
//!   temporal kernel splits each event's `±1` between the two time bins either side of `t*ᵢ`, in
//!   proportion to its closeness. Nothing is lost: every event lands whole, so the grid sums to
//!   `Σ pᵢ` — and to `N` with every event taken as ON — [`voxel_grid`]. (The CVPR 2019 version
//!   cited here prints `t₁` in both places, as the range it states, `[0, B − 1]`, requires. The
//!   arXiv preprint, arXiv:1812.08156v1, prints `t₀` in the numerator and `t₁` in the denominator,
//!   for events numbered from 1; earlier releases of this module attributed that slip to the CVPR
//!   version.)
//! - **Event frames in time bins**: per pixel and polarity, the number of events in each of `n`
//!   windows — [`time_frames`] over windows you choose, [`spanning_frames`] over the recording's
//!   own span with its last event included.
//!
//! [`crate::vision`] already accumulates a single frame and a time surface; this module adds the
//! time axis.
//!
//! # Against tonic, and what tonic does
//!
//! tonic (Lenz, Chaney, Shrestha, Oubari, Picaud and Zarrella, *Tonic: event-based datasets and
//! transformations*, Zenodo, 2021, doi:10.5281/zenodo.5079802) loads the public event datasets and
//! builds these tensors from them in Python.
//! The tests hold tonic 1.6.0's own output, and reproduce it — which is how four behaviours of it
//! were pinned down, none of them shared here:
//!
//! ⚠ **Its voxel grid scales to `[0, B]`, not `[0, B − 1]`.** `to_voxel_grid_numpy` multiplies by
//! `n_time_bins` where the paper it cites multiplies by `B − 1`, so the last event lands at `t* = B`
//! and is dropped, and every event in the final bin loses the share that belongs past it. Its grid
//! is exactly this module's grid with ONE MORE bin, the last one discarded — the tests reproduce it
//! that way, value for value. Five events of polarity `+1, −1, +1, +1, +1` sum to 2 in tonic's grid,
//! not 3.
//!
//! ⚠ **With tonic's own dataset dtype its voxel grid ignores polarity.** The function maps an OFF
//! event to `−1` with `pols[pols == 0] = -1`, and DVS Gesture's events, as tonic itself declares
//! them, store polarity as `bool`: assigning `−1` to a `bool` writes `True`. Every OFF event then
//! adds `+1`. Through the `ToVoxelGrid` transform the result is the same.
//!
//! ⚠ **Called directly, the same line rewrites the caller's array**: the polarity field of the
//! events passed in comes back with its zeros replaced. (The transform passes a copy.)
//!
//! ⚠ **Its time-binned frames drop the recording's last event, and count in `int16`.**
//! `to_frame_numpy(n_time_bins = n)` sizes each window as the duration integer-divided by `n` and
//! slices half-open windows from the first event, so the last event — and, when the duration does
//! not divide, the whole remainder — falls outside every frame: six events over 1 000 µs in three
//! bins give frames holding four. And 40 000 events on one pixel in one frame come back as
//! `−25 537`. [`spanning_frames`] closes its last frame on the last event; counts are `u64`.
//!
//! # Time
//!
//! Seconds, as everywhere in [`crate::vision`]; events must be in time order, which that module's
//! [`crate::vision::require_time_ordered`] checks.

use crate::spike::Polarity;
use crate::vision::{Geometry, PixelEvent, VisionError, require_time_ordered};

/// A voxel grid: `bins` time slices of a sensor, each pixel holding the polarity mass that fell
/// near that slice.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelGrid {
    /// The sensor.
    pub geom: Geometry,
    /// The number of time bins `B`.
    pub bins: usize,
    /// Values, bin-major: index `(b · height + y) · width + x`.
    pub values: Vec<f64>,
}

impl VoxelGrid {
    /// The value at bin `b`, pixel `(x, y)`; `None` outside the grid.
    #[must_use]
    pub fn at(&self, b: usize, x: u16, y: u16) -> Option<f64> {
        let pixel = self.geom.index(x, y)?;
        (b < self.bins).then(|| self.values[b * self.geom.pixels() + pixel])
    }

    /// The sum of every value: `Σ pᵢ`, up to rounding, since every event lands whole.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }
}

fn sign(p: Polarity) -> f64 {
    match p {
        Polarity::On => 1.0,
        Polarity::Off => -1.0,
    }
}

/// Zhu et al.'s discretized event volume of `events` over `bins` time bins.
///
/// The first event's time maps to bin 0 and the last's to bin `B − 1`; between them each event's
/// `±1` is split as `1 − δ` to bin `⌊t*⌋` and `δ` to the next, `δ = t* − ⌊t*⌋`. The last event has
/// `δ = 0` and lands whole in bin `B − 1`.
///
/// # Errors
///
/// [`VisionError::TooFew`] for fewer than two bins or no events; [`VisionError::OutOfOrder`] or
/// [`VisionError::NonFinite`] for a stream that is not in time order; [`VisionError::OutOfBounds`]
/// for an event off the sensor; [`VisionError::Degenerate`] when every event has the same time,
/// where equation (1) divides by zero.
pub fn voxel_grid(geom: Geometry, events: &[PixelEvent], bins: usize) -> Result<VoxelGrid, VisionError> {
    if bins < 2 {
        return Err(VisionError::TooFew { what: "time bins", have: bins, need: 2 });
    }
    let (Some(first), Some(last)) = (events.first(), events.last()) else {
        return Err(VisionError::TooFew { what: "events", have: 0, need: 1 });
    };
    require_time_ordered(events)?;
    let span = last.t_s - first.t_s;
    if !(span > 0.0) {
        return Err(VisionError::Degenerate { what: "event span, which equation (1) divides by" });
    }
    let plane = geom.pixels();
    let mut values = vec![0.0; bins * plane];
    let top = (bins - 1) as f64;
    for e in events {
        let pixel = geom.require(e.x, e.y)?;
        // For the last event `t*` is `top·span/span`, which rounds to one ulp past `B − 1` about as
        // often as not; it is the top of the range by definition, so it is clamped there.
        let t = (top * (e.t_s - first.t_s) / span).min(top);
        let lower = t.floor() as usize;
        let delta = t - lower as f64;
        let p = sign(e.polarity);
        values[lower * plane + pixel] += p * (1.0 - delta);
        if delta > 0.0 {
            values[(lower + 1) * plane + pixel] += p * delta;
        }
    }
    Ok(VoxelGrid { geom, bins, values })
}

/// Event counts in consecutive time windows, per polarity and pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeFrames {
    /// The sensor.
    pub geom: Geometry,
    /// The number of frames.
    pub frames: usize,
    /// Counts, frame-major then polarity (`Off` = 0, `On` = 1): index
    /// `((k · 2 + polarity) · height + y) · width + x`.
    pub counts: Vec<u64>,
    /// Events that fell in no frame.
    pub outside: usize,
}

impl TimeFrames {
    /// The count in frame `k` at pixel `(x, y)` for `polarity`; `None` outside the frames.
    #[must_use]
    pub fn at(&self, k: usize, polarity: Polarity, x: u16, y: u16) -> Option<u64> {
        let pixel = self.geom.index(x, y)?;
        let plane = usize::from(polarity == Polarity::On);
        (k < self.frames).then(|| self.counts[(k * 2 + plane) * self.geom.pixels() + pixel])
    }

    /// Every event counted.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// Frames `[start + k·window, start + (k + 1)·window)` for `k < n`, each event placed by comparing
/// its time with those computed edges.
///
/// # Errors
///
/// [`VisionError::NonPositive`] or [`VisionError::NonFinite`] for a bad `window` or `start`;
/// [`VisionError::TooFew`] for no frames; the stream and geometry errors of [`voxel_grid`].
pub fn time_frames(geom: Geometry, events: &[PixelEvent], start: f64, window: f64, n: usize) -> Result<TimeFrames, VisionError> {
    if n == 0 {
        return Err(VisionError::TooFew { what: "frames", have: 0, need: 1 });
    }
    if !start.is_finite() {
        return Err(VisionError::NonFinite { what: "start", value: start });
    }
    if !window.is_finite() {
        return Err(VisionError::NonFinite { what: "window", value: window });
    }
    if !(window > 0.0) {
        return Err(VisionError::NonPositive { what: "window", value: window });
    }
    let end = start + n as f64 * window;
    frames_by(geom, events, n, |t| (t >= start && t < end).then(|| frame_of(t, start, window, n)))
}

/// The frame `k < n` whose computed edges `start + k·window ≤ t < start + (k + 1)·window` hold `t`,
/// for a `t` at or after `start`: division's guess, corrected against the edges and never moved
/// outside the frames.
fn frame_of(t: f64, start: f64, window: f64, n: usize) -> usize {
    let edge = |k: usize| start + k as f64 * window;
    let mut k = (((t - start) / window) as usize).min(n - 1);
    while k > 0 && edge(k) > t {
        k -= 1;
    }
    while k + 1 < n && edge(k + 1) <= t {
        k += 1;
    }
    k
}

/// `n` frames of equal length spanning the stream from its first event to its last, the last frame
/// closed so that the last event is in it.
///
/// # Errors
///
/// As [`time_frames`], and [`VisionError::TooFew`] for no events and [`VisionError::Degenerate`]
/// when every event has the same time.
pub fn spanning_frames(geom: Geometry, events: &[PixelEvent], n: usize) -> Result<TimeFrames, VisionError> {
    if n == 0 {
        return Err(VisionError::TooFew { what: "frames", have: 0, need: 1 });
    }
    let (Some(first), Some(last)) = (events.first(), events.last()) else {
        return Err(VisionError::TooFew { what: "events", have: 0, need: 1 });
    };
    require_time_ordered(events)?;
    let span = last.t_s - first.t_s;
    if !(span > 0.0) {
        return Err(VisionError::Degenerate { what: "event span the frames divide" });
    }
    let window = span / n as f64;
    // Every event is in `[first, last]`, so every event is in a frame: the last frame is closed
    // because the search never moves past it.
    frames_by(geom, events, n, |t| Some(frame_of(t, first.t_s, window, n)))
}

fn frames_by(geom: Geometry, events: &[PixelEvent], n: usize, which: impl Fn(f64) -> Option<usize>) -> Result<TimeFrames, VisionError> {
    require_time_ordered(events)?;
    let plane = geom.pixels();
    let mut counts = vec![0u64; n * 2 * plane];
    let mut outside = 0;
    for e in events {
        let pixel = geom.require(e.x, e.y)?;
        match which(e.t_s) {
            Some(k) => counts[(k * 2 + usize::from(e.polarity == Polarity::On)) * plane + pixel] += 1,
            None => outside += 1,
        }
    }
    Ok(TimeFrames { geom, frames: n, counts, outside })
}

#[cfg(test)]
mod tests {
    use super::{spanning_frames, time_frames, voxel_grid};
    use crate::spike::Polarity;
    use crate::vision::{Geometry, PixelEvent, VisionError};

    /// `(t µs, x, y, on)`: forty random events on a 4 × 3 sensor.
    const EVENTS: [(u64, u16, u16, bool); 40] = [
        (20063, 0, 2, false),
        (100681, 0, 2, false),
        (105313, 0, 1, true),
        (107259, 0, 1, true),
        (146962, 0, 1, false),
        (162597, 2, 1, false),
        (338916, 2, 2, false),
        (364022, 0, 0, true),
        (383671, 2, 2, true),
        (386742, 2, 1, true),
        (407266, 1, 0, true),
        (408344, 1, 1, false),
        (408446, 1, 0, true),
        (429527, 3, 0, false),
        (457398, 2, 2, true),
        (466742, 1, 0, false),
        (467890, 0, 0, false),
        (496973, 1, 2, true),
        (506985, 0, 1, false),
        (537792, 3, 1, false),
        (545030, 0, 1, false),
        (550437, 3, 1, false),
        (646487, 0, 1, false),
        (745997, 0, 0, false),
        (750747, 1, 2, true),
        (756882, 1, 1, true),
        (770472, 2, 1, true),
        (781049, 2, 2, false),
        (831130, 3, 0, false),
        (832621, 1, 0, true),
        (835769, 3, 0, true),
        (836397, 0, 2, false),
        (846255, 0, 2, false),
        (847219, 3, 2, true),
        (869699, 0, 1, true),
        (884148, 3, 2, true),
        (891162, 1, 2, true),
        (912715, 1, 1, true),
        (916139, 0, 2, true),
        (966525, 1, 0, true),
    ];
    /// tonic 1.6.0 `to_voxel_grid_numpy(events, (4, 3, 2), n_time_bins = 5)`, flattened `[t][y][x]`.
    const TONIC_VOXEL: [f64; 60] = [0.0, 0.0, 0.0, 0.0, 0.7593828384023869, 0.0, -0.2470167846147019, 0.0, -1.574108627710357, 0.0, 0.0, 0.0, 0.18292229376351088, 0.0, 0.0, 0.0, 0.24061716159761304, 0.0, -0.6900868708939186, 0.0, -0.4258913722896429, 0.0, -0.23643315843636614, 0.0, 0.1828726351401322, 1.262439485156298, 0.0, -0.8368703656353875, -0.6543601327892721, -0.9487766017019172, 0.9371036555086205, -0.46304764480771565, 0.0, 0.4805644600628445, 0.9260657057546946, 0.0, -0.5308073646908169, -0.2624394851562979, 0.0, -0.16312963436461247, -2.0363469426136493, 0.05628540818331862, 0.03571511587364329, -1.5369523551922843, 0.0, 0.6593545224213968, 0.31036745268167154, 0.0, -0.8349875642128262, 0.7073923728580755, 0.0, -0.024507058920485214, 0.20222259319444413, 1.1767603981987658, 0.9642848841263567, 0.0, -1.056629848847603, 1.2582111062039472, -0.9798385989083558, 1.065457461577961];
    /// The same events with DVS Gesture's dtype (`p` is `bool`), through `tonic.transforms.ToVoxelGrid`.
    const TONIC_VOXEL_BOOL: [f64; 60] = [0.0, 0.0, 0.0, 0.0, 1.4186105728492004, 0.0, 0.2470167846147019, 0.0, 1.574108627710357, 0.0, 0.0, 0.0, 0.18292229376351088, 0.0, 0.0, 0.0, 1.5813894271507993, 0.0, 0.8158795598766776, 0.0, 0.4258913722896429, 0.0, 0.3946730032478851, 0.0, 1.451282777332846, 2.542979010250808, 0.0, 0.8368703656353875, 0.6543601327892721, 0.9487766017019172, 0.9371036555086205, 0.46304764480771565, 0.0, 0.4805644600628445, 2.2949595440704433, 0.0, 0.5308073646908169, 0.4570209897491919, 0.0, 0.16312963436461247, 2.0363469426136493, 0.15873220477948413, 0.03571511587364329, 1.5369523551922843, 0.0, 0.6593545224213968, 0.31036745268167154, 0.0, 0.8349875642128262, 0.7073923728580755, 0.0, 1.406031092637634, 0.8208084423886013, 1.1767603981987658, 0.9642848841263567, 0.0, 1.5889914227935193, 1.2582111062039472, 0.9798385989083558, 1.065457461577961];
    /// tonic 1.6.0 `to_frame_numpy(events, (4, 3, 2), n_time_bins = 3)`, flattened `[frame][p][y][x]`; its window is 946462 // 3 = 315487 µs.
    const TONIC_FRAMES: [i64; 72] = [0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 0, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 1, 1, 2, 1, 0, 1, 2, 0, 2];
    const TONIC_WINDOW_US: u64 = 315487;

    fn events(all_on: bool) -> Vec<PixelEvent> {
        EVENTS
            .iter()
            .map(|&(t, x, y, on)| PixelEvent { t_s: t as f64 * 1e-6, x, y, polarity: if on || all_on { Polarity::On } else { Polarity::Off } })
            .collect()
    }

    fn geom() -> Geometry {
        Geometry::new(4, 3).unwrap()
    }

    /// tonic's voxel grid is this module's grid with one more bin, the last one discarded — value for
    /// value, for forty random events — because it scales time to `[0, B]` where the paper scales to
    /// `[0, B − 1]`.
    #[test]
    fn tonics_voxel_grid_is_this_grid_with_one_more_bin() {
        let ours = voxel_grid(geom(), &events(false), 6).unwrap();
        for (k, (&a, &b)) in ours.values[..60].iter().zip(&TONIC_VOXEL).enumerate() {
            assert!((a - b).abs() <= 1e-12, "value {k}: {a} against tonic's {b}");
        }
        // What that costs: the grid the paper defines holds every event whole; tonic's does not.
        let paper = voxel_grid(geom(), &events(false), 5).unwrap();
        let signed: f64 = EVENTS.iter().map(|e| if e.3 { 1.0 } else { -1.0 }).sum();
        assert!((paper.sum() - signed).abs() < 1e-12, "{} against {signed}", paper.sum());
        let on = voxel_grid(geom(), &events(true), 5).unwrap();
        assert!((on.sum() - 40.0).abs() < 1e-12, "forty events, forty units of mass: {}", on.sum());
        let tonic_sum: f64 = TONIC_VOXEL.iter().sum();
        assert!((tonic_sum - signed).abs() > 0.5, "tonic's grid has lost mass: {tonic_sum} against {signed}");
        assert!((tonic_sum - ours.values[..60].iter().sum::<f64>()).abs() < 1e-12);
    }

    /// With DVS Gesture's own dtype tonic's voxel grid ignores polarity: its output is this module's
    /// grid (with one more bin) of the same events all taken as ON.
    #[test]
    fn with_a_bool_polarity_tonic_counts_every_event_as_on() {
        let all_on = voxel_grid(geom(), &events(true), 6).unwrap();
        for (k, (&a, &b)) in all_on.values[..60].iter().zip(&TONIC_VOXEL_BOOL).enumerate() {
            assert!((a - b).abs() <= 1e-12, "value {k}: {a} against tonic's {b}");
        }
        assert!(TONIC_VOXEL_BOOL.iter().all(|&v| v >= 0.0) && TONIC_VOXEL.iter().any(|&v| v < 0.0));
    }

    /// tonic's time-binned frames are this module's frames over its integer-divided window, and they
    /// miss events that the spanning frames count.
    #[test]
    fn tonics_frames_drop_the_tail_that_spanning_frames_keep() {
        let ev = events(false);
        let start = ev[0].t_s;
        let window = TONIC_WINDOW_US as f64 * 1e-6;
        let f = time_frames(geom(), &ev, start, window, 3).unwrap();
        let tonic: Vec<u64> = TONIC_FRAMES.iter().map(|&c| c as u64).collect();
        assert_eq!(f.counts, tonic);
        assert_eq!(f.total() + f.outside as u64, 40);
        assert!(f.outside >= 1, "the last event is past tonic's last window");
        let all = spanning_frames(geom(), &ev, 3).unwrap();
        assert_eq!((all.total(), all.outside), (40, 0));
        let last = ev[39];
        assert_eq!(all.at(2, last.polarity, last.x, last.y).map(|c| c >= 1), Some(true));
    }

    /// Every event lands whole, where the paper says it does: the first in bin 0, the last in bin
    /// `B − 1`, one exactly between two bin centres split in half.
    #[test]
    fn each_event_lands_whole_where_equation_one_puts_it() {
        let g = Geometry::new(2, 1).unwrap();
        let e = |t_s: f64, x: u16, polarity| PixelEvent { t_s, x, y: 0, polarity };
        // Times in binary fractions so t* is exact: B = 5, span 1, t* = 4t.
        let ev = [e(0.0, 0, Polarity::On), e(0.375, 1, Polarity::Off), e(1.0, 0, Polarity::On)];
        let v = voxel_grid(g, &ev, 5).unwrap();
        assert_eq!(v.at(0, 0, 0), Some(1.0), "the first event, whole, in bin 0");
        assert_eq!(v.at(4, 0, 0), Some(1.0), "the last event, whole, in bin B − 1");
        assert_eq!((v.at(1, 1, 0), v.at(2, 1, 0)), (Some(-0.5), Some(-0.5)), "t* = 1.5: half each side");
        assert_eq!(v.sum(), 1.0);
        assert_eq!((v.at(5, 0, 0), v.at(0, 2, 0)), (None, None));
        assert_eq!(v.values.len(), 10);
    }

    /// A frame count cannot wrap: forty thousand events on one pixel are forty thousand. (tonic's
    /// `int16` frames return −25 537 for them.)
    #[test]
    fn forty_thousand_events_are_forty_thousand() {
        let g = Geometry::new(1, 1).unwrap();
        let ev: Vec<PixelEvent> = (0..40_000).map(|k| PixelEvent { t_s: f64::from(k) * 1e-6, x: 0, y: 0, polarity: Polarity::On }).collect();
        let f = spanning_frames(g, &ev, 1).unwrap();
        assert_eq!(f.at(0, Polarity::On, 0, 0), Some(40_000));
        assert_eq!(f.at(0, Polarity::Off, 0, 0), Some(0));
        assert_eq!(f.at(1, Polarity::On, 0, 0), None);
    }

    /// Frames place each event by the computed edges `start + k·window`, and what is outside them is
    /// counted.
    #[test]
    fn frames_place_events_by_their_computed_edges() {
        let g = Geometry::new(1, 1).unwrap();
        let e = |t_s: f64| PixelEvent { t_s, x: 0, y: 0, polarity: Polarity::Off };
        // 1.46 IS the edge 1.3 + 16 × 0.01, though (1.46 − 1.3)/0.01 is 15.99…
        let f = time_frames(g, &[e(1.2), e(1.46), e(1.5)], 1.3, 0.01, 20).unwrap();
        assert_eq!(f.at(16, Polarity::Off, 0, 0), Some(1));
        assert_eq!(f.outside, 2, "1.2 is before the first frame and 1.5 after the last");
        // 10.999999999999998 is below the edge 33 × (1/3), though it divides to 33.
        let f = time_frames(g, &[e(10.999_999_999_999_998)], 0.0, 1.0 / 3.0, 40).unwrap();
        assert_eq!(f.at(32, Polarity::Off, 0, 0), Some(1));
    }

    /// Every refusal.
    #[test]
    fn every_refusal_names_what_it_refused() {
        let g = geom();
        let on = |t_s: f64, x: u16| PixelEvent { t_s, x, y: 0, polarity: Polarity::On };
        let two = [on(0.0, 0), on(1.0, 1)];
        assert_eq!(voxel_grid(g, &two, 1), Err(VisionError::TooFew { what: "time bins", have: 1, need: 2 }));
        assert_eq!(voxel_grid(g, &[], 5), Err(VisionError::TooFew { what: "events", have: 0, need: 1 }));
        assert_eq!(voxel_grid(g, &[on(0.5, 0), on(0.5, 1)], 5), Err(VisionError::Degenerate { what: "event span, which equation (1) divides by" }));
        assert!(matches!(voxel_grid(g, &[on(1.0, 0), on(0.5, 1)], 5), Err(VisionError::OutOfOrder { .. })));
        assert!(matches!(voxel_grid(g, &[on(0.0, 0), on(1.0, 9)], 5), Err(VisionError::OutOfBounds { x: 9, .. })));
        assert!(matches!(time_frames(g, &two, f64::NAN, 0.1, 3), Err(VisionError::NonFinite { what: "start", .. })));
        assert!(matches!(time_frames(g, &two, 0.0, f64::INFINITY, 3), Err(VisionError::NonFinite { what: "window", .. })));
        assert_eq!(time_frames(g, &two, 0.0, 0.0, 3), Err(VisionError::NonPositive { what: "window", value: 0.0 }));
        assert_eq!(time_frames(g, &two, 0.0, 0.1, 0), Err(VisionError::TooFew { what: "frames", have: 0, need: 1 }));
        assert!(matches!(time_frames(g, &[on(1.0, 0), on(0.5, 1)], 0.0, 0.1, 3), Err(VisionError::OutOfOrder { .. })));
        assert!(matches!(time_frames(g, &[on(0.0, 9)], 0.0, 0.1, 3), Err(VisionError::OutOfBounds { .. })));
        assert_eq!(spanning_frames(g, &[], 3), Err(VisionError::TooFew { what: "events", have: 0, need: 1 }));
        assert_eq!(spanning_frames(g, &[on(0.5, 0), on(0.5, 1)], 3), Err(VisionError::Degenerate { what: "event span the frames divide" }));
        assert_eq!(spanning_frames(g, &two, 0), Err(VisionError::TooFew { what: "frames", have: 0, need: 1 }));
        assert!(matches!(spanning_frames(g, &[on(1.0, 0), on(0.5, 1)], 3), Err(VisionError::OutOfOrder { .. })));
    }
}
