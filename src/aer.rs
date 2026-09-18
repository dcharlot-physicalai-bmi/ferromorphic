//! Address-event representation: the wire formats event sensors actually speak.
//!
//! # What `AER` is, and what it buys
//!
//! A conventional image sensor is read out on a clock: every pixel reports every frame, whether or
//! not anything happened to it. Misha Mahowald's address-event representation (doctoral thesis,
//! *VLSI Phototransduction and Stereopsis*, Caltech, 1992; and Sivilotti's contemporaneous work at
//! the same lab) replaces that with a **shared digital bus that carries the identity of whichever
//! element just fired**. A 128x128 array does not need 16,384 wires; it needs 14 address lines and
//! an arbiter. The spike itself is not transmitted — its *address* is, and its *time* is the moment
//! the transmission happens.
//!
//! That buys three things. Nothing is sent when nothing changes, so a static scene costs nothing.
//! Latency is set by the bus, not by a frame period, so an event reaches the receiver in
//! microseconds. And the bus cost scales with activity rather than with pixel count, which is the
//! whole economic argument for event sensing.
//!
//! It costs three things, and this module exists because of the third. Two elements firing at once
//! must be serialised by an arbiter, so simultaneity is destroyed at the source. The timestamp is a
//! property of the *receiver*, not of the pixel, so it inherits the bus's queueing jitter. And —
//! the part a library can actually fix — **every vendor invented a different way of writing the
//! stream to a file**, and all of them compress time by sending it once per group of events rather
//! than once per event.
//!
//! # Why the formats are state machines, and why that is the bug farm
//!
//! A modern event camera emits 10^5 to 10^8 events per second. At a naive 8 bytes per event that is
//! up to 800 MB/s, so every format in the field shares one timestamp across many events: a
//! `TIME_HIGH` word sets the upper bits, and the individual events carry only the low bits. The
//! decoder is therefore a **state machine that reconstructs an absolute time from a running base**,
//! and it has the two failure modes state machines always have.
//!
//! *Rollover.* The low bits wrap. `EVT` 3.0 puts 12 bits of time on an event and 12 more in a
//! `TIME_HIGH` word, so the wire carries 24 bits — 16.777216 seconds — and a recording longer than
//! that wraps repeatedly. A decoder that does not count the wraps produces a sawtooth timestamp
//! that still looks monotonic within each 16-second window, so a plot of the first second looks
//! perfect. [`Evt3`] counts them, and
//! `evt3_reconstructs_timestamps_across_repeated_rollovers` is the test that says so.
//!
//! *Lost base.* If the 12-bit low time *decreases* without an intervening `TIME_HIGH`, the high
//! word was dropped and the decoder must advance the base itself. The classic bug is the mirror
//! image: forgetting to reset the remembered low value when a `TIME_HIGH` *does* arrive, so the
//! first small low value in the new window is mistaken for a wrap and time jumps 4.096 ms forward.
//! Both branches are exercised by `evt3_recovers_a_dropped_time_high_word` and
//! `evt3_does_not_invent_a_wrap_after_a_time_high`.
//!
//! # What every decoder here guarantees
//!
//! 1. **Total.** No input panics. Not a truncated file, not random bytes, not a header without a
//!    terminator. `every_truncation_of_every_format_errors_instead_of_panicking` decodes every
//!    prefix of a valid stream, byte by byte, for all six formats; `random_bytes_never_panic`
//!    throws 4,000 seeded random buffers at them. A decoder that panics on a short read is a
//!    decoder that crashes a robot mid-flight, and the file being short is the *normal* case when
//!    a recording is interrupted.
//! 2. **Errors carry the byte offset.** [`DecodeError`] names where and what, because "invalid
//!    file" is not an actionable message for a 4 GB recording.
//! 3. **Timestamps come out monotonically non-decreasing, or the call fails.** Non-decreasing
//!    rather than increasing: the arbiter genuinely emits several events in the same microsecond,
//!    and collapsing them would be wrong. A stream whose reconstruction would go backwards returns
//!    [`DecodeError::NonMonotonicTimestamp`] instead of a plausible event list.
//! 4. **Encode-decode is bit-exact** for every format, over thousands of events spanning the full
//!    coordinate and timestamp range. That is what makes the decoders checkable at all: there is no
//!    published reference vector for most of these formats, so the encoder is the oracle, and the
//!    encoder is in turn pinned to the published bit layouts by hand-computed word tests.
//!
//! # Units, and the one place they are not SI
//!
//! Every timestamp in this module is **microseconds**, as a `u64`, because that is what all four
//! vendor formats put on the wire and converting at the boundary would make the hand-checkable
//! word tests unreadable. [`TrainMap`] converts to the crate's tick convention
//! ([`crate::spike::Spike::t`]) at the point of use, once, with the tick length stated.
//! Coordinates are pixels: `x` to the right, `y` down, origin top-left, which is the convention all
//! four formats use on the wire. Whether a given sensor's *optics* invert that is a property of the
//! camera, not of the file, and this module does not attempt to correct it.
//!
//! # A worked example
//!
//! ```
//! use ferromorphic::aer::{AerEvent, Evt3, TrainMap};
//! use ferromorphic::spike::Polarity;
//!
//! // Three adjacent pixels of a horizontal edge in one microsecond, then one 5 ms later.
//! let events = vec![
//!     AerEvent { t: 1_000, x: 320, y: 240, polarity: Polarity::On },
//!     AerEvent { t: 1_000, x: 321, y: 240, polarity: Polarity::On },
//!     AerEvent { t: 1_000, x: 322, y: 240, polarity: Polarity::On },
//!     AerEvent { t: 6_000, x: 100, y: 12, polarity: Polarity::Off },
//! ];
//!
//! // Vectorised, the run of three becomes a base column plus one 8-bit mask.
//! let wire = Evt3::encode(&events, true)?;
//! assert_eq!(Evt3::decode(&wire)?.events, events);
//!
//! // Into this crate's tick-indexed spike train: one tick per millisecond, On and Off events
//! // given separate address planes so the contrast direction survives.
//! let map = TrainMap { width: 640, height: 480, tick_us: 1_000, split_polarity: true };
//! let train = map.map(&events).expect("inside the stated geometry");
//! assert_eq!(train.len(), 4);
//! assert_eq!(train.spikes()[0].t, 1);
//! assert_eq!(train.spikes()[0].source, 240 * 640 + 320 + 640 * 480);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Provenance, and where this implementation is unsure
//!
//! `AEDAT` 2.0 and 4.0 are the `jAER` / `DV` formats from the Institute of Neuroinformatics,
//! Zurich (`jAER` project, Delbruck et al., 2007 onward). `EVT` 2.0, `EVT` 3.0 and the older `.dat`
//! are `Prophesee`'s, documented in the `Metavision` SDK. None of these has an ISO specification; the
//! bit layouts below are transcribed from the vendors' public documentation and from open-source
//! readers, and **every place where this implementation could not confirm a convention says so in
//! the doc of the item concerned** rather than presenting a guess confidently. The three known soft
//! spots are the polarity bit's sense in `AEDAT` 2.0 ([`Aedat2Layout::p_on_is_one`]), the `.dat`
//! record-type byte ([`Dat::CD_TYPE_CODES`]), and the `FlatBuffers` table layout inside an
//! `AEDAT` 4.0 packet ([`Aedat4`]), which this implementation did not check against a file produced
//! by `DV` software.

use crate::spike::{Event, Polarity, Train};

// ---------------------------------------------------------------------------------------------
// The common event
// ---------------------------------------------------------------------------------------------

/// One decoded change-detection event: a pixel crossed its contrast threshold.
///
/// Field order is `(t, x, y, polarity)` so that the derived `Ord` sorts by time first, which is the
/// order every format requires on the wire and every consumer wants in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AerEvent {
    /// Timestamp in **microseconds** since the recording's zero, as the file expresses it.
    ///
    /// Not seconds: every format on the wire counts microseconds, and a float conversion here would
    /// put a rounding between the decoder and the bit-exactness test that validates it.
    pub t: u64,
    /// Column, pixels, origin at the left edge. Range is the format's field width — 11 bits for
    /// `EVT` 2.0 and `EVT` 3.0, 14 bits for `.dat`, layout-dependent for `AEDAT` 2.0.
    pub x: u16,
    /// Row, pixels, origin at the **top** edge. Same range rules as [`AerEvent::x`].
    pub y: u16,
    /// Sign of the contrast change: [`Polarity::On`] for brighter, [`Polarity::Off`] for darker.
    pub polarity: Polarity,
}

impl AerEvent {
    /// Flatten to the crate's [`crate::spike::Event`], addressing as `y * width + x`.
    ///
    /// Returns `None` when `x` is at or past `width`, or when the flattened address would not fit
    /// in a `u32`. Both are refusals rather than clamps: an out-of-range `x` means the caller's
    /// `width` disagrees with the sensor that wrote the file, and folding it into the next row
    /// produces a picture that is sheared rather than obviously wrong.
    ///
    /// The tick index is the microsecond count unchanged, i.e. a 1 microsecond tick. Use
    /// [`TrainMap`] to choose a different one.
    #[must_use]
    pub fn to_event(self, width: u16) -> Option<Event> {
        if width == 0 || self.x >= width {
            return None;
        }
        let address =
            u32::from(self.y).checked_mul(u32::from(width))?.checked_add(u32::from(self.x))?;
        Some(Event { t: self.t, address, polarity: self.polarity })
    }

    /// Unflatten a [`crate::spike::Event`] whose address is `y * width + x`.
    ///
    /// Returns `None` for `width == 0`, or when the recovered row or column exceeds a `u16` — which
    /// no real sensor reaches, and which a synthetic address can.
    #[must_use]
    pub fn from_event(e: Event, width: u16) -> Option<Self> {
        if width == 0 {
            return None;
        }
        let w = u32::from(width);
        let y = u16::try_from(e.address / w).ok()?;
        let x = u16::try_from(e.address % w).ok()?;
        Some(Self { t: e.t, x, y, polarity: e.polarity })
    }
}

/// A word the decoder recognised as well-formed but did not interpret as a pixel event.
///
/// External trigger pulses, vendor "other" words and multi-word continuations are reported rather
/// than dropped, because a trigger is usually the only thing tying an event stream to whatever else
/// the robot was recording, and a decoder that silently swallows it destroys the synchronisation
/// evidence while leaving the event count unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Marker {
    /// Byte offset of the word in the input buffer, so a caller can go and look at it.
    pub offset: usize,
    /// What kind of non-pixel word this was.
    pub kind: MarkerKind,
    /// The raw word, zero-extended to 64 bits, exactly as it appeared on the wire.
    pub raw: u64,
    /// Reconstructed timestamp in microseconds at the moment this word was seen.
    ///
    /// Uses whatever time base the decoder held at that point, so a marker before the first
    /// `TIME_HIGH` word reads zero. That is the honest answer: the stream did not say.
    pub t: u64,
}

/// Which family of non-pixel word a [`Marker`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MarkerKind {
    /// An external trigger pulse — `EXT_TRIGGER` in `Prophesee`'s formats, a special-address event
    /// in `AEDAT` 2.0. This is the wire's synchronisation channel.
    ExternalTrigger,
    /// A vendor-defined `OTHERS` word: monitoring, padding, or a sensor-specific status code whose
    /// payload layout this implementation did not locate documentation for.
    Other,
    /// A `CONTINUED` word carrying the tail of a multi-word payload begun by the preceding marker.
    Continued,
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a stream could not be decoded. Every variant names a **byte offset** into the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The input ended in the middle of something.
    Truncated {
        /// Byte offset where the incomplete item starts.
        offset: usize,
        /// Bytes the item needs.
        need: usize,
        /// Bytes actually left from `offset` to the end of the input.
        have: usize,
    },
    /// The file did not begin with the magic this decoder requires.
    ///
    /// Raised eagerly and deliberately: feeding an `EVT` 3.0 recording to the `AEDAT` 2.0 decoder
    /// otherwise produces millions of plausible events at plausible coordinates, because every
    /// 8-byte group of a dense event stream is a valid `AEDAT` 2.0 record.
    BadMagic {
        /// Byte offset of the first mismatching byte.
        offset: usize,
        /// The magic this decoder expected, as text.
        expected: &'static str,
        /// What was actually there, lossily converted, truncated to the expected length.
        found: String,
    },
    /// A word's opcode field holds a value the format does not define.
    BadOpcode {
        /// Byte offset of the word.
        offset: usize,
        /// The undefined opcode, right-aligned in a byte.
        code: u8,
    },
    /// A field held a value outside the range its width or the file's own header allows.
    FieldOutOfRange {
        /// Byte offset of the word or record the field belongs to.
        offset: usize,
        /// Which field, named as the format's documentation names it.
        field: &'static str,
        /// The offending value.
        value: u64,
        /// The largest value the field may hold.
        max: u64,
    },
    /// Reconstructing the timestamp would have gone backwards.
    ///
    /// Almost always a dropped or corrupted time word rather than a sensor fault. Reported instead
    /// of repaired because the repair is a guess about how much time was lost, and a downstream
    /// tracker fed a silently guessed interval reports a confident wrong velocity.
    NonMonotonicTimestamp {
        /// Byte offset of the word that produced the decrease.
        offset: usize,
        /// The timestamp before it, microseconds.
        previous: u64,
        /// The timestamp it would have produced, microseconds.
        found: u64,
    },
    /// An `EVT` 3.0 `EVT_ADDR_X` or `VECT_BASE_X` word arrived before any `EVT_ADDR_Y` word.
    ///
    /// The format sends the row once and the columns many times, so a column word with no row in
    /// hand is a stream that was cut mid-row. This is the expected first error when a recording is
    /// resumed from an arbitrary offset.
    ColumnBeforeRow {
        /// Byte offset of the column word.
        offset: usize,
    },
    /// An `EVT` 3.0 `VECT_12` or `VECT_8` mask arrived with no `VECT_BASE_X` to anchor it.
    VectorBeforeBase {
        /// Byte offset of the mask word.
        offset: usize,
    },
    /// An `EVT` 3.0 `EVT_ADDR_Y` word set the master/slave bit.
    ///
    /// That bit marks a second camera's events multiplexed into one stream. This implementation
    /// refuses rather than merging two cameras' pixels into a single coordinate space, which would
    /// produce a full, sorted, entirely wrong event list.
    UnsupportedSystemType {
        /// Byte offset of the row word.
        offset: usize,
    },
    /// An `AEDAT` 4.0 packet is compressed, and this crate has no decompressor.
    ///
    /// Stated as a capability boundary, not a file defect: [`Aedat4::decode`] still returns the
    /// full packet framing, and only [`Aedat4::events`] refuses. See the note on [`Aedat4`].
    UnsupportedCompression {
        /// Byte offset of the packet header.
        offset: usize,
        /// The `Format:` value read from the file header.
        name: String,
    },
    /// A container's declared item count disagrees with the bytes present.
    CountMismatch {
        /// Byte offset of the count field.
        offset: usize,
        /// The count the file declared.
        declared: u64,
        /// The count the remaining bytes can actually supply.
        actual: u64,
    },
    /// A byte reserved for a future version of a format was not zero.
    ///
    /// Only [`Flat`], this crate's own interchange format, enforces this. Reserved bytes that are
    /// allowed to hold rubbish cannot be reclaimed later, so they are checked from the first
    /// version rather than from the version that needs them.
    ReservedNotZero {
        /// Byte offset of the offending byte.
        offset: usize,
    },
    /// The record size or type byte of a `.dat` file is one this implementation does not handle.
    UnsupportedRecordLayout {
        /// Byte offset of the two-byte type/size pair.
        offset: usize,
        /// The record type byte.
        record_type: u8,
        /// The record size byte, in bytes per record.
        record_size: u8,
    },
    /// An [`Aedat2Layout`]'s bit fields overlap, so no single address can encode both.
    ///
    /// A configuration error rather than a file error, caught before any byte is read because the
    /// alternative is a decode that succeeds and puts the polarity bit inside the column.
    LayoutFieldsOverlap {
        /// The bits claimed by more than one field.
        mask: u32,
    },
    /// An `AEDAT` 4.0 packet payload is not a `FlatBuffers` table this implementation can walk.
    MalformedFlatBuffer {
        /// Byte offset of the packet payload's first byte within the whole file.
        offset: usize,
        /// What was wrong, in the `FlatBuffers` vocabulary: `root offset`, `vtable`, `vector`.
        what: &'static str,
    },
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { offset, need, have } => write!(
                f,
                "input ends at byte {offset}: {need} bytes needed for the next item, {have} available"
            ),
            Self::BadMagic { offset, expected, found } => write!(
                f,
                "byte {offset}: expected the magic {expected:?}, found {found:?}"
            ),
            Self::BadOpcode { offset, code } => {
                write!(f, "byte {offset}: opcode 0x{code:X} is not defined by this format")
            }
            Self::FieldOutOfRange { offset, field, value, max } => {
                write!(f, "byte {offset}: {field} is {value}, which exceeds its maximum {max}")
            }
            Self::NonMonotonicTimestamp { offset, previous, found } => write!(
                f,
                "byte {offset}: timestamp would go backwards, from {previous} us to {found} us; a time word was probably lost"
            ),
            Self::ColumnBeforeRow { offset } => write!(
                f,
                "byte {offset}: a column word arrived before any row word, so the stream was cut mid-row"
            ),
            Self::VectorBeforeBase { offset } => {
                write!(f, "byte {offset}: a vector mask arrived with no VECT_BASE_X to anchor it")
            }
            Self::UnsupportedSystemType { offset } => write!(
                f,
                "byte {offset}: the row word sets the master/slave bit; this decoder will not merge two cameras into one coordinate space"
            ),
            Self::UnsupportedCompression { offset, name } => write!(
                f,
                "byte {offset}: packet payload is {name}, and this crate carries no decompressor; the framing decoded, the events did not"
            ),
            Self::CountMismatch { offset, declared, actual } => {
                write!(f, "byte {offset}: {declared} items declared, {actual} present")
            }
            Self::ReservedNotZero { offset } => {
                write!(f, "byte {offset}: a reserved byte is not zero")
            }
            Self::UnsupportedRecordLayout { offset, record_type, record_size } => write!(
                f,
                "byte {offset}: record type 0x{record_type:02X} of {record_size} bytes is not a layout this decoder handles"
            ),
            Self::LayoutFieldsOverlap { mask } => {
                write!(f, "address layout fields overlap on bits 0x{mask:08X}")
            }
            Self::MalformedFlatBuffer { offset, what } => {
                write!(f, "byte {offset}: FlatBuffers {what} is out of bounds or inconsistent")
            }
        }
    }
}

/// See the note on [`crate::net::NetError`]: a library error has to be able to cross a
/// `Box<dyn Error>` boundary or its callers reach for `.unwrap()`.
impl std::error::Error for DecodeError {}

/// Why a stream could not be encoded. Every variant names the **index of the event** at fault.
///
/// Encoding fails for exactly one reason: the caller's data does not fit the format. That is
/// information — it says the recording cannot be expressed in the format asked for — so it is an
/// error rather than a silent truncation of the high bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// A field of an event does not fit the format's bit width or the sensor geometry given.
    FieldOutOfRange {
        /// Index into the caller's event slice.
        index: usize,
        /// Which field, named as the format's documentation names it.
        field: &'static str,
        /// The offending value.
        value: u64,
        /// The largest value the field may hold.
        max: u64,
    },
    /// The events are not sorted by timestamp.
    ///
    /// Every format here shares one time base across many events, so an out-of-order event cannot
    /// be written at all — not merely written inefficiently. Sorting on the caller's behalf would
    /// hide a bug in whatever produced the order.
    Unsorted {
        /// Index of the first event that is earlier than its predecessor.
        index: usize,
        /// The predecessor's timestamp, microseconds.
        previous: u64,
        /// This event's timestamp, microseconds.
        found: u64,
    },
    /// A header line contains a newline, which would silently split it into two lines on decode.
    HeaderLineContainsNewline {
        /// Index into the caller's header slice.
        index: usize,
    },
    /// A gap between consecutive events exceeds what the format's on-wire time field can express.
    ///
    /// `EVT` 3.0 carries 24 bits of time, so it can express a gap up to 16.777215 s and no more;
    /// beyond that the number of wraps is simply not on the wire and no decoder can recover it.
    GapTooLarge {
        /// Index of the later event.
        index: usize,
        /// The gap in microseconds.
        gap: u64,
        /// The largest gap the format can express, microseconds.
        max: u64,
    },
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FieldOutOfRange { index, field, value, max } => {
                write!(f, "event {index}: {field} is {value}, which exceeds this format's maximum {max}")
            }
            Self::Unsorted { index, previous, found } => write!(
                f,
                "event {index} is at {found} us, before event {} at {previous} us; these formats share a time base and cannot express that",
                index.saturating_sub(1)
            ),
            Self::HeaderLineContainsNewline { index } => {
                write!(f, "header line {index} contains a newline and would not survive a round trip")
            }
            Self::GapTooLarge { index, gap, max } => write!(
                f,
                "event {index} is {gap} us after its predecessor; this format can express at most {max} us between events"
            ),
        }
    }
}

/// So `?` works in a caller whose error type is `Box<dyn Error>`, as every example here does.
impl std::error::Error for EncodeError {}

// ---------------------------------------------------------------------------------------------
// Byte helpers. Every one of them is total.
// ---------------------------------------------------------------------------------------------

fn slice_at(b: &[u8], at: usize, n: usize) -> Result<&[u8], DecodeError> {
    let end = at.checked_add(n).ok_or(DecodeError::Truncated { offset: at, need: n, have: 0 })?;
    b.get(at..end).ok_or(DecodeError::Truncated {
        offset: at,
        need: n,
        have: b.len().saturating_sub(at),
    })
}

fn u16_le(b: &[u8], at: usize) -> Result<u16, DecodeError> {
    let s = slice_at(b, at, 2)?;
    Ok(u16::from(s[0]) | (u16::from(s[1]) << 8))
}

fn u32_le(b: &[u8], at: usize) -> Result<u32, DecodeError> {
    let s = slice_at(b, at, 4)?;
    Ok(u32::from(s[0])
        | (u32::from(s[1]) << 8)
        | (u32::from(s[2]) << 16)
        | (u32::from(s[3]) << 24))
}

fn u64_le(b: &[u8], at: usize) -> Result<u64, DecodeError> {
    let lo = u64::from(u32_le(b, at)?);
    let hi = u64::from(u32_le(b, at + 4)?);
    Ok(lo | (hi << 32))
}

fn u32_be(b: &[u8], at: usize) -> Result<u32, DecodeError> {
    let s = slice_at(b, at, 4)?;
    Ok((u32::from(s[0]) << 24)
        | (u32::from(s[1]) << 16)
        | (u32::from(s[2]) << 8)
        | u32::from(s[3]))
}

/// Read `#`- or `%`-prefixed ASCII header lines from the front of a buffer.
///
/// Returns the lines with the prefix and the line terminator removed, and the offset of the first
/// byte after the header. A header line that runs to the end of the buffer with no newline is
/// [`DecodeError::Truncated`] rather than a silently accepted final line, because the missing
/// newline is exactly what a cut-short file looks like.
fn read_header_lines(b: &[u8], prefix: u8) -> Result<(Vec<String>, usize), DecodeError> {
    let mut lines = Vec::new();
    let mut at = 0usize;
    while b.get(at) == Some(&prefix) {
        let start = at + 1;
        let nl = b[start..].iter().position(|&c| c == b'\n').ok_or(DecodeError::Truncated {
            offset: at,
            need: b.len() - at + 1,
            have: b.len() - at,
        })?;
        let mut end = start + nl;
        let text_end = if end > start && b[end - 1] == b'\r' { end - 1 } else { end };
        lines.push(String::from_utf8_lossy(&b[start..text_end]).into_owned());
        end += 1;
        at = end;
    }
    Ok((lines, at))
}

fn push_header_lines(
    out: &mut Vec<u8>,
    lines: &[String],
    prefix: u8,
    crlf: bool,
) -> Result<(), EncodeError> {
    for (i, line) in lines.iter().enumerate() {
        if line.contains('\n') || line.contains('\r') {
            return Err(EncodeError::HeaderLineContainsNewline { index: i });
        }
        out.push(prefix);
        out.extend_from_slice(line.as_bytes());
        if crlf {
            out.push(b'\r');
        }
        out.push(b'\n');
    }
    Ok(())
}

fn check_sorted(events: &[AerEvent]) -> Result<(), EncodeError> {
    for i in 1..events.len() {
        if events[i].t < events[i - 1].t {
            return Err(EncodeError::Unsorted {
                index: i,
                previous: events[i - 1].t,
                found: events[i].t,
            });
        }
    }
    Ok(())
}

fn fit(index: usize, field: &'static str, value: u64, max: u64) -> Result<(), EncodeError> {
    if value > max {
        return Err(EncodeError::FieldOutOfRange { index, field, value, max });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// AEDAT 2.0 — the jAER format
// ---------------------------------------------------------------------------------------------

/// How a sensor's `(x, y, polarity)` is packed into the 32-bit address of an `AEDAT` 2.0 record.
///
/// `AEDAT` 2.0 does not describe its own address layout: the file says which chip wrote it in a
/// header comment, and the reader is expected to know that chip's bit assignment. That is why this
/// is a **parameter of the decoder** rather than something read from the file, and why passing the
/// wrong one produces a full, sorted, plausible, wrong event list rather than an error.
///
/// Presets are given for the two common cases, with their provenance and their uncertainty stated
/// on each constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aedat2Layout {
    /// Bit position of the least significant column bit.
    pub x_shift: u32,
    /// Number of column bits. The column field holds `0 ..= 2^x_bits - 1`.
    pub x_bits: u32,
    /// Whether the column counts from the right edge, i.e. stored value is `2^x_bits - 1 - x`.
    ///
    /// Several `AER` chips wire the column address backwards relative to the optical image. The
    /// transform is its own inverse, so an encode-decode round trip cannot detect a wrong setting;
    /// only a picture of a known scene can.
    pub x_invert: bool,
    /// Bit position of the least significant row bit.
    pub y_shift: u32,
    /// Number of row bits.
    pub y_bits: u32,
    /// Whether the row counts from the bottom edge. Same caveat as [`Aedat2Layout::x_invert`].
    pub y_invert: bool,
    /// Bit position of the single polarity bit.
    pub p_shift: u32,
    /// Whether a set polarity bit means [`Polarity::On`].
    ///
    /// **This is the flag this implementation is least sure of.** Third-party readers of the same
    /// chips disagree about the sense of this bit, and both choices produce an event stream that
    /// looks entirely normal — the scene simply has its contrast inverted, which no event count,
    /// rate plot or timestamp check can detect. It is exposed as a field, and each preset states
    /// what this implementation chose, so that a caller who has a recording of a known stimulus can
    /// settle it for their own data rather than inheriting a guess.
    pub p_on_is_one: bool,
    /// Bits that mark a non-pixel word. A record whose address has any of these bits set becomes a
    /// [`Marker`] instead of an event. Zero disables the check.
    pub special_mask: u32,
    /// Sensor width in pixels; a decoded column at or past it is [`DecodeError::FieldOutOfRange`].
    /// Zero disables the check, which is the right setting for an unknown chip.
    pub width: u16,
    /// Sensor height in pixels; same rule as [`Aedat2Layout::width`].
    pub height: u16,
    /// Which chip this layout describes, for an error message and for a recorded provenance.
    pub source: &'static str,
}

impl Aedat2Layout {
    /// The `DVS128` retina (Lichtsteiner, Posch and Delbruck, *IEEE J. Solid-State Circuits*
    /// 43(2):566-576, 2008): 128x128, 15 address bits.
    ///
    /// Row in bits 8-14, column in bits 1-7, polarity in bit 0, column counted from the right.
    /// Transcribed from the `jAER` `Tmpdiff128` extractor as this implementation reads it.
    /// **Unverified against a physical device**: the polarity sense here is "bit 0 clear means
    /// `On`", and the opposite appears in circulating `Python` readers. See
    /// [`Aedat2Layout::p_on_is_one`].
    pub const DVS128: Self = Self {
        x_shift: 1,
        x_bits: 7,
        x_invert: true,
        y_shift: 8,
        y_bits: 7,
        y_invert: false,
        p_shift: 0,
        p_on_is_one: false,
        special_mask: 0,
        width: 128,
        height: 128,
        source: "DVS128 (Lichtsteiner et al. 2008), jAER Tmpdiff128 extractor",
    };

    /// The `DAVIS346` (Taverni et al., *IEEE Trans. Circuits Syst. II* 65(5), 2018): 346x260.
    ///
    /// Row in bits 22-30, column in bits 12-21, polarity in bit 11, bit 31 marking a special or
    /// frame-sample word. Transcribed from the `jAER` `DavisBaseCamera` constants as this
    /// implementation reads them; **this implementation did not verify them against a `DAVIS`
    /// recording**, and in particular the interleaved active-pixel-sensor samples that share this
    /// address space are reported as [`MarkerKind::Other`] rather than decoded.
    pub const DAVIS346: Self = Self {
        x_shift: 12,
        x_bits: 10,
        x_invert: false,
        y_shift: 22,
        y_bits: 9,
        y_invert: false,
        p_shift: 11,
        p_on_is_one: true,
        special_mask: 1 << 31,
        width: 346,
        height: 260,
        source: "DAVIS346 (Taverni et al. 2018), jAER DavisBaseCamera constants",
    };

    /// Bits claimed by more than one field, or `0` if the layout is consistent.
    ///
    /// Checked before every decode and encode. An overlapping layout is a caller bug that would
    /// otherwise succeed: the polarity bit would land inside the column field and the image would
    /// be interleaved with itself.
    #[must_use]
    pub fn overlap(&self) -> u32 {
        let x = Self::field_mask(self.x_shift, self.x_bits);
        let y = Self::field_mask(self.y_shift, self.y_bits);
        let p = Self::field_mask(self.p_shift, 1);
        (x & y) | (x & p) | (y & p) | (x & self.special_mask) | (y & self.special_mask)
            | (p & self.special_mask)
    }

    fn field_mask(shift: u32, bits: u32) -> u32 {
        if bits == 0 || shift >= 32 {
            return 0;
        }
        let bits = bits.min(32 - shift);
        let m = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        m << shift
    }

    fn check(&self) -> Result<(), DecodeError> {
        let mask = self.overlap();
        if mask != 0 {
            return Err(DecodeError::LayoutFieldsOverlap { mask });
        }
        Ok(())
    }

    fn get(&self, addr: u32, shift: u32, bits: u32, invert: bool) -> u16 {
        let m = Self::field_mask(shift, bits) >> shift;
        let raw = (addr >> shift) & m;
        let v = if invert { m - raw } else { raw };
        // `m` is at most 2^bits - 1 with bits <= 31 here for any layout that passed `check`; the
        // truncation is therefore impossible for x_bits or y_bits of 16 or fewer, and for a wider
        // field the saturation is preferable to a wrap.
        u16::try_from(v).unwrap_or(u16::MAX)
    }

    fn put(&self, v: u16, shift: u32, bits: u32, invert: bool) -> u32 {
        let m = Self::field_mask(shift, bits) >> shift;
        let raw = if invert { m.saturating_sub(u32::from(v)) } else { u32::from(v) };
        (raw & m) << shift
    }
}

/// A decoded `AEDAT` 2.0 file.
///
/// The format is an ASCII header of `#`-prefixed lines followed by a flat array of **8-byte
/// big-endian records**: a 32-bit address then a 32-bit timestamp in microseconds. Big-endian
/// because `jAER` is written in `Java` and `Java`'s `DataOutputStream` is big-endian; this is the
/// only format in this module that is not little-endian, and it is the single most common way to
/// misread an `AEDAT` 2.0 file.
///
/// # The timestamp wraps, and this decoder unwraps it
///
/// `jAER` writes the timestamp as a signed 32-bit microsecond counter, so it goes negative after
/// 35.8 minutes and wraps to zero after 71.6. Read as an **unsigned** 32-bit pattern the signed
/// overflow disappears and the counter is simply monotonic to 2^32, which is what this decoder
/// does; a further wrap past 2^32 adds 2^32 to an accumulator. A *small* decrease is not treated as
/// a wrap — it is [`DecodeError::NonMonotonicTimestamp`], because the alternative is to add 71.6
/// minutes to the rest of the recording on the strength of one jittered record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat2 {
    /// Header lines with their leading `#` and trailing `\r\n` removed, first line included. For a
    /// `jAER` file the first is `!AER-DAT2.0`.
    pub header: Vec<String>,
    /// Events, in file order, with timestamps unwrapped to a monotonic microsecond count.
    pub events: Vec<AerEvent>,
    /// Non-pixel records: those whose address matched [`Aedat2Layout::special_mask`].
    pub markers: Vec<Marker>,
    /// The layout the file was decoded with, recorded so that a result cannot be separated from the
    /// assumption that produced it.
    pub layout: Aedat2Layout,
}

impl Aedat2 {
    /// The magic the first header line must start with.
    pub const MAGIC: &'static str = "!AER-DAT";

    /// Decode a whole `AEDAT` 2.0 file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::LayoutFieldsOverlap`] if `layout` is self-contradictory;
    /// [`DecodeError::BadMagic`] if the first line is not `#!AER-DAT...`, which is what catches a
    /// file of another format being fed in; [`DecodeError::Truncated`] if the header has no
    /// terminator or the record array ends mid-record; [`DecodeError::FieldOutOfRange`] if a
    /// decoded coordinate is outside the layout's stated sensor geometry;
    /// [`DecodeError::NonMonotonicTimestamp`] if the timestamps go backwards by less than the wrap.
    pub fn decode(bytes: &[u8], layout: Aedat2Layout) -> Result<Self, DecodeError> {
        layout.check()?;
        let (header, mut at) = read_header_lines(bytes, b'#')?;
        match header.first() {
            Some(first) if first.starts_with(Self::MAGIC) => {}
            _ => {
                let n = Self::MAGIC.len() + 1;
                return Err(DecodeError::BadMagic {
                    offset: 0,
                    expected: "#!AER-DAT2.0",
                    found: String::from_utf8_lossy(&bytes[..bytes.len().min(n)]).into_owned(),
                });
            }
        }

        let mut events = Vec::new();
        let mut markers = Vec::new();
        let mut wrap = 0u64;
        let mut prev_raw: Option<u32> = None;
        let mut prev_t = 0u64;
        while at < bytes.len() {
            let addr = u32_be(bytes, at)?;
            let raw_t = u32_be(bytes, at + 4)?;
            // 2^31 as the wrap threshold, i.e. "the counter jumped more than half its range
            // backwards". Anything smaller is a disordered file, not a wrap.
            if let Some(p) = prev_raw
                && raw_t < p
            {
                if p - raw_t > 1u32 << 31 {
                    wrap += 1u64 << 32;
                } else {
                    return Err(DecodeError::NonMonotonicTimestamp {
                        offset: at + 4,
                        previous: prev_t,
                        found: wrap + u64::from(raw_t),
                    });
                }
            }
            prev_raw = Some(raw_t);
            let t = wrap + u64::from(raw_t);
            prev_t = t;

            if layout.special_mask != 0 && addr & layout.special_mask != 0 {
                markers.push(Marker {
                    offset: at,
                    kind: MarkerKind::Other,
                    raw: u64::from(addr),
                    t,
                });
                at += 8;
                continue;
            }

            let x = layout.get(addr, layout.x_shift, layout.x_bits, layout.x_invert);
            let y = layout.get(addr, layout.y_shift, layout.y_bits, layout.y_invert);
            if layout.width != 0 && x >= layout.width {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at,
                    field: "column",
                    value: u64::from(x),
                    max: u64::from(layout.width - 1),
                });
            }
            if layout.height != 0 && y >= layout.height {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at,
                    field: "row",
                    value: u64::from(y),
                    max: u64::from(layout.height - 1),
                });
            }
            let bit = (addr >> layout.p_shift) & 1 == 1;
            let polarity =
                if bit == layout.p_on_is_one { Polarity::On } else { Polarity::Off };
            events.push(AerEvent { t, x, y, polarity });
            at += 8;
        }
        Ok(Self { header, events, markers, layout })
    }

    /// Encode events into an `AEDAT` 2.0 file.
    ///
    /// `header` lines are written with a leading `#` and a `\r\n` terminator, `jAER`'s convention;
    /// if the first line does not already start with [`Aedat2::MAGIC`], `!AER-DAT2.0` is prepended
    /// so that the result is decodable by [`Aedat2::decode`] and by `jAER`.
    ///
    /// Timestamps are written modulo 2^32 microseconds, which is what the format holds. A recording
    /// longer than 71.6 minutes therefore relies on the decoder's unwrapping to come back intact,
    /// and it does — that is what `aedat2_unwraps_a_timestamp_across_the_32_bit_wrap` checks.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order;
    /// [`EncodeError::FieldOutOfRange`] if a coordinate does not fit the layout's field width or
    /// its sensor geometry; [`EncodeError::HeaderLineContainsNewline`] if a header line would not
    /// survive the round trip.
    ///
    /// # Panics
    ///
    /// Does not panic. The layout check that would otherwise be a panic is folded into
    /// [`EncodeError::FieldOutOfRange`] on the `layout` field, named `overlapping bit fields`.
    pub fn encode(
        events: &[AerEvent],
        layout: Aedat2Layout,
        header: &[String],
    ) -> Result<Vec<u8>, EncodeError> {
        if layout.overlap() != 0 {
            return Err(EncodeError::FieldOutOfRange {
                index: 0,
                field: "overlapping bit fields in the layout",
                value: u64::from(layout.overlap()),
                max: 0,
            });
        }
        check_sorted(events)?;
        let mut out = Vec::with_capacity(64 + events.len() * 8);
        let mut lines: Vec<String> = Vec::new();
        if !header.first().is_some_and(|l| l.starts_with(Self::MAGIC)) {
            lines.push("!AER-DAT2.0".to_string());
        }
        lines.extend_from_slice(header);
        push_header_lines(&mut out, &lines, b'#', true)?;

        let x_max = u64::from(Aedat2Layout::field_mask(layout.x_shift, layout.x_bits) >> layout.x_shift);
        let y_max = u64::from(Aedat2Layout::field_mask(layout.y_shift, layout.y_bits) >> layout.y_shift);
        for (i, e) in events.iter().enumerate() {
            let x_lim = if layout.width == 0 { x_max } else { x_max.min(u64::from(layout.width) - 1) };
            let y_lim =
                if layout.height == 0 { y_max } else { y_max.min(u64::from(layout.height) - 1) };
            fit(i, "column", u64::from(e.x), x_lim)?;
            fit(i, "row", u64::from(e.y), y_lim)?;
            let mut addr = layout.put(e.x, layout.x_shift, layout.x_bits, layout.x_invert)
                | layout.put(e.y, layout.y_shift, layout.y_bits, layout.y_invert);
            let on = e.polarity == Polarity::On;
            if on == layout.p_on_is_one {
                addr |= 1u32 << layout.p_shift;
            }
            let ts = (e.t & 0xFFFF_FFFF) as u32;
            out.extend_from_slice(&addr.to_be_bytes());
            out.extend_from_slice(&ts.to_be_bytes());
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// EVT 2.0 — Prophesee, fixed 32-bit words
// ---------------------------------------------------------------------------------------------

/// A decoded `EVT` 2.0 stream: `Prophesee`'s fixed-width format.
///
/// Every word is 32 bits little-endian, with a 4-bit opcode in bits 31-28. A change-detection word
/// is laid out
///
/// ```text
///  31    28 27      22 21        11 10         0
/// [ type  ][ time low ][  column   ][   row     ]
///     4         6           11           11
/// ```
///
/// and the opcode is `0x0` for a darker event and `0x1` for a brighter one, so the opcode *is* the
/// polarity for pixel words. The 6-bit time field is the low bits of a microsecond counter whose
/// upper 28 bits arrive in a separate `EVT_TIME_HIGH` word, giving 34 bits of time on the wire —
/// 4 hours 46 minutes before it wraps, which is long enough that this decoder's wrap counter is
/// mostly theoretical and is tested anyway.
///
/// Fixed width makes this the easy one: 8 bytes of information in 4, no state except the time base,
/// and a decoder that cannot lose sync because every word boundary is at a multiple of four. The
/// cost is that it spends 22 bits per event on a coordinate pair that barely changes between
/// neighbouring events, which is exactly the redundancy [`Evt3`] removes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Evt2 {
    /// Events in file order, timestamps reconstructed to absolute microseconds.
    pub events: Vec<AerEvent>,
    /// Trigger, `OTHERS` and `CONTINUED` words, reported but not interpreted.
    pub markers: Vec<Marker>,
}

impl Evt2 {
    /// Opcode of a change-detection event with falling contrast.
    pub const CD_OFF: u8 = 0x0;
    /// Opcode of a change-detection event with rising contrast.
    pub const CD_ON: u8 = 0x1;
    /// Opcode of the word carrying the upper 28 bits of the microsecond counter.
    pub const TIME_HIGH: u8 = 0x8;
    /// Opcode of an external trigger pulse.
    pub const EXT_TRIGGER: u8 = 0xA;
    /// Opcode of a vendor-defined monitoring word.
    pub const OTHERS: u8 = 0xE;
    /// Opcode of a continuation word extending the previous one.
    pub const CONTINUED: u8 = 0xF;
    /// Largest timestamp the wire can express: 34 bits of microseconds, minus one.
    pub const MAX_TIME_US: u64 = (1u64 << 34) - 1;

    /// Decode an `EVT` 2.0 stream.
    ///
    /// The stream must be a whole number of 4-byte words; a trailing fragment is
    /// [`DecodeError::Truncated`] naming its offset, which is what a recording cut mid-word by a
    /// power loss looks like.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] on a partial final word; [`DecodeError::BadOpcode`] for any of
    /// the ten opcodes `EVT` 2.0 leaves undefined, with the offset of the word;
    /// [`DecodeError::NonMonotonicTimestamp`] if the 6-bit low time decreases without an
    /// intervening `EVT_TIME_HIGH`, which means a time word was lost and this decoder will not
    /// guess how much time went with it.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut events = Vec::new();
        let mut markers = Vec::new();
        let mut base = 0u64;
        let mut high = 0u32;
        let mut wraps = 0u64;
        let mut last_t = 0u64;
        let mut at = 0usize;
        while at < bytes.len() {
            let w = u32_le(bytes, at)?;
            let code = (w >> 28) as u8;
            let low = u64::from((w >> 22) & 0x3F);
            match code {
                Self::CD_OFF | Self::CD_ON => {
                    let t = base + low;
                    if t < last_t {
                        return Err(DecodeError::NonMonotonicTimestamp {
                            offset: at,
                            previous: last_t,
                            found: t,
                        });
                    }
                    last_t = t;
                    let x = ((w >> 11) & 0x7FF) as u16;
                    let y = (w & 0x7FF) as u16;
                    let polarity =
                        if code == Self::CD_ON { Polarity::On } else { Polarity::Off };
                    events.push(AerEvent { t, x, y, polarity });
                }
                Self::TIME_HIGH => {
                    let h = w & 0x0FFF_FFFF;
                    if h < high {
                        wraps += 1;
                    }
                    high = h;
                    base = (wraps << 34) | (u64::from(high) << 6);
                    // Deliberately NOT compared against the last event's time. A sensor emits a
                    // TIME_HIGH every 64 us whether or not the value changed, so a repeat carrying
                    // the value already held is normal and must not be refused; the base it sets
                    // is below the last event's time by up to 63 us by construction. Time going
                    // backwards is caught where it matters, on the next event.
                }
                Self::EXT_TRIGGER | Self::OTHERS | Self::CONTINUED => {
                    let kind = match code {
                        Self::EXT_TRIGGER => MarkerKind::ExternalTrigger,
                        Self::OTHERS => MarkerKind::Other,
                        _ => MarkerKind::Continued,
                    };
                    markers.push(Marker { offset: at, kind, raw: u64::from(w), t: base + low });
                }
                _ => return Err(DecodeError::BadOpcode { offset: at, code }),
            }
            at += 4;
        }
        Ok(Self { events, markers })
    }

    /// Encode events as `EVT` 2.0.
    ///
    /// An `EVT_TIME_HIGH` word is emitted whenever the upper 28 bits of the timestamp change, and
    /// once before the first event whatever its time, so that a decoder joining the stream at byte
    /// zero has a base. Real sensors emit one every 64 microseconds regardless; this encoder emits
    /// the minimum, which decodes identically and is what makes the round trip a test of the
    /// decoder rather than of a padding convention.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order, and
    /// [`EncodeError::FieldOutOfRange`] if a timestamp exceeds [`Evt2::MAX_TIME_US`] or a
    /// coordinate exceeds 2047 — the format's 11-bit fields, which no `Prophesee` sensor exceeds
    /// and a synthetic stream easily can.
    pub fn encode(events: &[AerEvent]) -> Result<Vec<u8>, EncodeError> {
        check_sorted(events)?;
        let mut out = Vec::with_capacity(8 + events.len() * 4);
        let mut last_high: Option<u32> = None;
        for (i, e) in events.iter().enumerate() {
            fit(i, "timestamp", e.t, Self::MAX_TIME_US)?;
            fit(i, "column", u64::from(e.x), 2047)?;
            fit(i, "row", u64::from(e.y), 2047)?;
            let high = ((e.t >> 6) & 0x0FFF_FFFF) as u32;
            if last_high != Some(high) {
                let w = (u32::from(Self::TIME_HIGH) << 28) | high;
                out.extend_from_slice(&w.to_le_bytes());
                last_high = Some(high);
            }
            let code = if e.polarity == Polarity::On { Self::CD_ON } else { Self::CD_OFF };
            let low = (e.t & 0x3F) as u32;
            let w = (u32::from(code) << 28)
                | (low << 22)
                | (u32::from(e.x) << 11)
                | u32::from(e.y);
            out.extend_from_slice(&w.to_le_bytes());
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// EVT 3.0 — Prophesee, variable-length and vectorised. The hard one.
// ---------------------------------------------------------------------------------------------

/// A decoded `EVT` 3.0 stream: `Prophesee`'s variable-length, vectorised format.
///
/// Words are **16 bits** little-endian, opcode in bits 15-12, payload in bits 11-0. Nothing is
/// repeated that has not changed: the row is sent once and reused by every column word that
/// follows, the time is sent once per microsecond that has events in it, and a horizontal run of
/// pixels is sent as a base column plus a **bitmask** of which of the next 8 or 12 columns fired.
/// A dense horizontal edge therefore costs about 1.3 bits per event instead of 32.
///
/// # The word table
///
/// ```text
/// 0x0 EVT_ADDR_Y      bit 11 = master/slave, bits 10-0 = row; sets the row for what follows
/// 0x2 EVT_ADDR_X      bit 11 = polarity,     bits 10-0 = column; emits one event
/// 0x3 VECT_BASE_X     bit 11 = polarity,     bits 10-0 = base column; emits NOTHING
/// 0x4 VECT_12         bits 11-0 = mask of 12 columns from the base; base += 12
/// 0x5 VECT_8          bits  7-0 = mask of  8 columns from the base; base += 8
/// 0x6 EVT_TIME_LOW    bits 11-0 = bits 11-0 of the microsecond counter
/// 0x7 CONTINUED_4     4-bit continuation of the preceding word
/// 0x8 EVT_TIME_HIGH   bits 11-0 = bits 23-12 of the microsecond counter
/// 0xA EXT_TRIGGER     external trigger pulse
/// 0xE OTHERS          vendor monitoring word
/// 0xF CONTINUED_12    12-bit continuation of the preceding word
/// ```
///
/// Opcodes `0x1`, `0x9`, `0xB`, `0xC` and `0xD` are not defined, and this decoder rejects them with
/// their offset rather than skipping them, because a stream that has drifted out of alignment
/// produces exactly those codes and skipping turns a detectable desync into a corrupted event list.
///
/// # Time on the wire is 24 bits, and that is a hard limit
///
/// `EVT_TIME_HIGH` carries bits 23-12, `EVT_TIME_LOW` carries bits 11-0. **There is no more.** Past
/// 2^24 microseconds — 16.777216 s — the counter wraps and the only evidence is that the new high
/// word is *smaller* than the old one. [`Evt3::decode`] counts those wraps into an unbounded `u64`,
/// which is why a decoded timestamp can exceed what the file could hold.
///
/// The consequence for the encoder is the interesting one: **a gap longer than the wrap cannot be
/// written at all**, because the number of whole wraps inside it is not on the wire and no decoder
/// could recover it. [`Evt3::encode`] returns [`EncodeError::GapTooLarge`] rather than writing a
/// stream that decodes to a different recording. Subtract the recording's start time before
/// encoding, too: the first timestamp must be below 2^24 for the same reason.
///
/// # The two rollover bugs, named
///
/// A low word that is *smaller* than the previous low word, with no `EVT_TIME_HIGH` in between,
/// means the high word was dropped; the decoder must advance the high itself. The mirror-image bug
/// is forgetting to clear the remembered low when an `EVT_TIME_HIGH` *does* arrive — the next low
/// then looks like a wrap and time leaps 4096 microseconds. Both are one line each and both produce
/// a perfectly monotonic, perfectly wrong timestamp, which is why they survive so long in the
/// field. A repeated `EVT_TIME_HIGH` carrying the *same* value does **not** clear the low, because
/// that would move time backwards inside one window.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Evt3 {
    /// Events in file order, timestamps reconstructed to absolute microseconds with wraps counted.
    pub events: Vec<AerEvent>,
    /// Trigger, `OTHERS` and `CONTINUED` words, reported but not interpreted.
    pub markers: Vec<Marker>,
}

impl Evt3 {
    /// Opcode: set the row for subsequent column words.
    pub const ADDR_Y: u16 = 0x0;
    /// Opcode: one event at this column, in the current row, at the current time.
    pub const ADDR_X: u16 = 0x2;
    /// Opcode: set the base column and polarity for the vector masks that follow.
    pub const VECT_BASE_X: u16 = 0x3;
    /// Opcode: 12-column bitmask from the current base.
    pub const VECT_12: u16 = 0x4;
    /// Opcode: 8-column bitmask from the current base.
    pub const VECT_8: u16 = 0x5;
    /// Opcode: bits 11-0 of the microsecond counter.
    pub const TIME_LOW: u16 = 0x6;
    /// Opcode: 4-bit continuation of the preceding word.
    pub const CONTINUED_4: u16 = 0x7;
    /// Opcode: bits 23-12 of the microsecond counter.
    pub const TIME_HIGH: u16 = 0x8;
    /// Opcode: external trigger pulse.
    pub const EXT_TRIGGER: u16 = 0xA;
    /// Opcode: vendor monitoring word.
    pub const OTHERS: u16 = 0xE;
    /// Opcode: 12-bit continuation of the preceding word.
    pub const CONTINUED_12: u16 = 0xF;
    /// Wrap period of the on-wire counter, microseconds: 2^24.
    pub const WRAP_US: u64 = 1 << 24;
    /// Largest first timestamp an encodable stream may have, microseconds.
    pub const MAX_FIRST_TIME_US: u64 = (1 << 24) - 1;
    /// A gap between consecutive events that is always encodable, microseconds: `4095 * 4096`.
    ///
    /// The exact condition the encoder tests is that the 12-bit high word advances by at most 4095
    /// steps, which depends on where inside a 4096-microsecond window the two events fall; a gap up
    /// to this constant satisfies it whatever the phase, and slightly larger gaps sometimes do.
    /// This is the bound that can be stated without knowing the phase, so it is the one reported.
    pub const MAX_GAP_US: u64 = 4095 * 4096;

    /// Decode an `EVT` 3.0 stream.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] on a trailing odd byte; [`DecodeError::BadOpcode`] for the five
    /// undefined opcodes; [`DecodeError::ColumnBeforeRow`] for a column word with no row in hand,
    /// which is what a stream resumed from an arbitrary offset produces first;
    /// [`DecodeError::VectorBeforeBase`] for a mask with no base; [`DecodeError::FieldOutOfRange`]
    /// if a mask bit implies a column past 2047; [`DecodeError::UnsupportedSystemType`] if the row
    /// word marks a second camera; [`DecodeError::NonMonotonicTimestamp`] if the reconstruction
    /// would go backwards, which the rules above make impossible and which is checked anyway so
    /// that a future edit to those rules fails loudly instead of quietly.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut events = Vec::new();
        let mut markers = Vec::new();
        let mut row: Option<u16> = None;
        let mut vect_base: Option<u16> = None;
        let mut vect_pol = Polarity::Off;
        let mut high = 0u32;
        let mut low = 0u32;
        let mut overflows = 0u64;
        let mut t = 0u64;
        let mut last_t = 0u64;
        let mut at = 0usize;

        while at < bytes.len() {
            let w = u16_le(bytes, at)?;
            let code = w >> 12;
            let payload = w & 0x0FFF;
            match code {
                Self::ADDR_Y => {
                    if payload & 0x800 != 0 {
                        return Err(DecodeError::UnsupportedSystemType { offset: at });
                    }
                    row = Some(payload & 0x7FF);
                }
                Self::ADDR_X => {
                    let Some(y) = row else {
                        return Err(DecodeError::ColumnBeforeRow { offset: at });
                    };
                    let polarity =
                        if payload & 0x800 != 0 { Polarity::On } else { Polarity::Off };
                    events.push(AerEvent { t, x: payload & 0x7FF, y, polarity });
                }
                Self::VECT_BASE_X => {
                    if row.is_none() {
                        return Err(DecodeError::ColumnBeforeRow { offset: at });
                    }
                    vect_pol = if payload & 0x800 != 0 { Polarity::On } else { Polarity::Off };
                    vect_base = Some(payload & 0x7FF);
                }
                Self::VECT_12 | Self::VECT_8 => {
                    let Some(y) = row else {
                        return Err(DecodeError::ColumnBeforeRow { offset: at });
                    };
                    let Some(base) = vect_base else {
                        return Err(DecodeError::VectorBeforeBase { offset: at });
                    };
                    let (width, mask) = if code == Self::VECT_12 {
                        (12u16, payload)
                    } else {
                        (8u16, payload & 0x00FF)
                    };
                    for k in 0..width {
                        if mask & (1 << k) == 0 {
                            continue;
                        }
                        let x = u32::from(base) + u32::from(k);
                        if x > 2047 {
                            return Err(DecodeError::FieldOutOfRange {
                                offset: at,
                                field: "column implied by a vector mask bit",
                                value: u64::from(x),
                                max: 2047,
                            });
                        }
                        events.push(AerEvent { t, x: x as u16, y, polarity: vect_pol });
                    }
                    // Saturating so that a malformed stream walking the base off the end of the
                    // row cannot wrap it back into valid coordinates; the next mask bit then
                    // fails the range check above instead of producing a plausible event.
                    vect_base = Some(base.saturating_add(width));
                }
                Self::TIME_LOW => {
                    let l = u32::from(payload);
                    if l < low {
                        // The high word was dropped. The wire cannot say how many were dropped,
                        // and one is what a single low-wrap is evidence for, so one is what this
                        // advances. Two consecutive missing high words are not recoverable and
                        // this decoder does not pretend otherwise.
                        high = (high + 1) & 0x0FFF;
                        if high == 0 {
                            overflows += 1;
                        }
                    }
                    low = l;
                    t = (overflows << 24) | (u64::from(high) << 12) | u64::from(low);
                }
                Self::TIME_HIGH => {
                    let h = u32::from(payload);
                    if h != high {
                        if h < high {
                            overflows += 1;
                        }
                        high = h;
                        // Cleared ONLY when the window actually changed. A repeated TIME_HIGH
                        // carrying the same value must not rewind time inside its own window.
                        low = 0;
                    }
                    t = (overflows << 24) | (u64::from(high) << 12) | u64::from(low);
                }
                Self::EXT_TRIGGER | Self::OTHERS | Self::CONTINUED_12 | Self::CONTINUED_4 => {
                    let kind = match code {
                        Self::EXT_TRIGGER => MarkerKind::ExternalTrigger,
                        Self::OTHERS => MarkerKind::Other,
                        _ => MarkerKind::Continued,
                    };
                    markers.push(Marker { offset: at, kind, raw: u64::from(w), t });
                }
                _ => {
                    return Err(DecodeError::BadOpcode { offset: at, code: code as u8 });
                }
            }
            if t < last_t {
                return Err(DecodeError::NonMonotonicTimestamp {
                    offset: at,
                    previous: last_t,
                    found: t,
                });
            }
            last_t = t;
            at += 2;
        }
        Ok(Self { events, markers })
    }

    /// Encode events as `EVT` 3.0.
    ///
    /// With `vectorise` set, a run of three or more events sharing a timestamp, a row and a
    /// polarity, with strictly increasing columns, is written as a `VECT_BASE_X` plus `VECT_12` and
    /// `VECT_8` masks. With it clear, every event gets its own `EVT_ADDR_X` word. **Both decode to
    /// the same events** — that is `evt3_vectorised_and_plain_encodings_decode_identically`, and it
    /// is the check that says the vector path is a compression rather than a second format.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order;
    /// [`EncodeError::FieldOutOfRange`] if the first timestamp is at or past
    /// [`Evt3::MAX_FIRST_TIME_US`] or a coordinate exceeds 2047; [`EncodeError::GapTooLarge`] if
    /// two consecutive events are far enough apart that the number of counter wraps between them is
    /// not on the wire.
    pub fn encode(events: &[AerEvent], vectorise: bool) -> Result<Vec<u8>, EncodeError> {
        check_sorted(events)?;
        let mut out: Vec<u8> = Vec::with_capacity(events.len() * 2 + 8);
        let mut cur_high: Option<u32> = None;
        let mut cur_low: Option<u32> = None;
        let mut cur_row: Option<u16> = None;

        let push = |out: &mut Vec<u8>, code: u16, payload: u16| {
            let w = (code << 12) | (payload & 0x0FFF);
            out.extend_from_slice(&w.to_le_bytes());
        };

        if let Some(first) = events.first() {
            fit(0, "first timestamp", first.t, Self::MAX_FIRST_TIME_US)?;
        }
        let mut i = 0usize;
        while i < events.len() {
            let e = events[i];
            fit(i, "column", u64::from(e.x), 2047)?;
            fit(i, "row", u64::from(e.y), 2047)?;
            if i > 0 {
                let prev = events[i - 1];
                let adv = (e.t >> 12) - (prev.t >> 12);
                if adv > 4095 {
                    return Err(EncodeError::GapTooLarge {
                        index: i,
                        gap: e.t - prev.t,
                        max: Self::MAX_GAP_US,
                    });
                }
            }
            let high = ((e.t >> 12) & 0x0FFF) as u32;
            let low = (e.t & 0x0FFF) as u32;
            if cur_high != Some(high) {
                push(&mut out, Self::TIME_HIGH, high as u16);
                cur_high = Some(high);
                cur_low = None;
            }
            if cur_low != Some(low) {
                push(&mut out, Self::TIME_LOW, low as u16);
                cur_low = Some(low);
            }
            if cur_row != Some(e.y) {
                push(&mut out, Self::ADDR_Y, e.y);
                cur_row = Some(e.y);
            }

            // How many events share this event's time, row and polarity with strictly increasing
            // columns? That is the run the vector encoding can express.
            let mut j = i + 1;
            if vectorise {
                while j < events.len()
                    && events[j].t == e.t
                    && events[j].y == e.y
                    && events[j].polarity == e.polarity
                    && events[j].x > events[j - 1].x
                    && events[j].x <= 2047
                {
                    j += 1;
                }
            }
            let run = &events[i..j];
            let pol_bit = if e.polarity == Polarity::On { 0x800u16 } else { 0 };
            if run.len() >= 3 {
                let mut base = run[0].x;
                push(&mut out, Self::VECT_BASE_X, base | pol_bit);
                let mut k = 0usize;
                while k < run.len() {
                    if run[k].x >= base.saturating_add(12) {
                        base = run[k].x;
                        push(&mut out, Self::VECT_BASE_X, base | pol_bit);
                    }
                    // An 8-wide window is enough exactly when everything still to come fits in it.
                    let use8 = run[run.len() - 1].x < base.saturating_add(8);
                    let width: u16 = if use8 { 8 } else { 12 };
                    let mut mask = 0u16;
                    while k < run.len() && run[k].x < base.saturating_add(width) {
                        mask |= 1 << (run[k].x - base);
                        k += 1;
                    }
                    push(&mut out, if use8 { Self::VECT_8 } else { Self::VECT_12 }, mask);
                    base = base.saturating_add(width);
                }
            } else {
                for ev in run {
                    let bit = if ev.polarity == Polarity::On { 0x800u16 } else { 0 };
                    push(&mut out, Self::ADDR_X, ev.x | bit);
                }
            }
            i = j;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// .dat — Prophesee's older format
// ---------------------------------------------------------------------------------------------

/// A decoded `Prophesee` `.dat` file: the format that preceded `EVT` 2.0.
///
/// A `%`-prefixed ASCII header, then a two-byte record descriptor — a type code and a size in bytes
/// — then a flat array of records. The change-detection record is 8 bytes little-endian:
///
/// ```text
/// bytes 0-3: uint32 timestamp, microseconds
/// bytes 4-7: uint32 payload, with
///            bits 13-0  column
///            bits 27-14 row
///            bits 31-28 polarity (only bit 28 is used)
/// ```
///
/// Fourteen bits per coordinate is generous to the point of waste for a 1280x720 sensor, and eight
/// bytes per event is twice `EVT` 2.0 and four times a vectorised `EVT` 3.0 — which is the whole
/// reason the newer formats exist. It survives because it is the easiest format in the field to
/// read: no state, no opcode, no reconstruction. A `NumPy` one-liner reads it, and that is why
/// most published event-vision datasets are still distributed in it.
///
/// The 32-bit microsecond timestamp wraps after 71.6 minutes; [`Dat::decode`] unwraps it the same
/// way [`Aedat2`] does, and refuses a small decrease rather than adding 71.6 minutes on the
/// strength of one record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dat {
    /// Header lines with their leading `%` and trailing newline removed.
    pub header: Vec<String>,
    /// The record type byte the file declared. See [`Dat::CD_TYPE_CODES`].
    pub record_type: u8,
    /// The record size byte the file declared, in bytes. This decoder handles 8.
    pub record_size: u8,
    /// Sensor width if a `% Width <n>` header line gave one, used to range-check columns.
    pub width: Option<u16>,
    /// Sensor height if a `% Height <n>` header line gave one, used to range-check rows.
    pub height: Option<u16>,
    /// Events in file order, timestamps unwrapped to a monotonic microsecond count.
    pub events: Vec<AerEvent>,
}

impl Dat {
    /// Record type bytes this decoder accepts as 8-byte change-detection records.
    ///
    /// `0x00` is the original `Event2D`; `0x0C` is what later `Prophesee` tooling writes for
    /// `EventCD`. **This implementation did not locate an authoritative table** and found both in
    /// circulation, so both are accepted and the byte is preserved in [`Dat::record_type`] so that
    /// a re-encode returns the file's own value rather than a normalised one. Any other type byte
    /// is [`DecodeError::UnsupportedRecordLayout`] rather than a guess.
    pub const CD_TYPE_CODES: [u8; 2] = [0x00, 0x0C];
    /// The record size in bytes this decoder handles.
    pub const RECORD_SIZE: u8 = 8;
    /// Largest column or row the 14-bit fields hold.
    pub const MAX_COORD: u64 = 16_383;

    fn header_number(header: &[String], key: &str) -> Option<u16> {
        for line in header {
            let mut it = line.split_whitespace();
            if it.next() == Some(key)
                && let Some(v) = it.next()
            {
                return v.parse::<u16>().ok();
            }
        }
        None
    }

    /// Decode a `.dat` file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] if the header has no terminator, the two-byte descriptor is
    /// missing, or the record array ends mid-record;
    /// [`DecodeError::UnsupportedRecordLayout`] if the descriptor is not an 8-byte change-detection
    /// record; [`DecodeError::FieldOutOfRange`] if a coordinate exceeds a `% Width` or `% Height`
    /// the file itself declared; [`DecodeError::NonMonotonicTimestamp`] if timestamps go backwards
    /// by less than a counter wrap.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (header, mut at) = read_header_lines(bytes, b'%')?;
        let desc = slice_at(bytes, at, 2)?;
        let (record_type, record_size) = (desc[0], desc[1]);
        if !Self::CD_TYPE_CODES.contains(&record_type) || record_size != Self::RECORD_SIZE {
            return Err(DecodeError::UnsupportedRecordLayout {
                offset: at,
                record_type,
                record_size,
            });
        }
        at += 2;
        let width = Self::header_number(&header, "Width");
        let height = Self::header_number(&header, "Height");

        let mut events = Vec::new();
        let mut wrap = 0u64;
        let mut prev_raw: Option<u32> = None;
        let mut prev_t = 0u64;
        while at < bytes.len() {
            let raw_t = u32_le(bytes, at)?;
            let data = u32_le(bytes, at + 4)?;
            if let Some(p) = prev_raw
                && raw_t < p
            {
                if p - raw_t > 1u32 << 31 {
                    wrap += 1u64 << 32;
                } else {
                    return Err(DecodeError::NonMonotonicTimestamp {
                        offset: at,
                        previous: prev_t,
                        found: wrap + u64::from(raw_t),
                    });
                }
            }
            prev_raw = Some(raw_t);
            let t = wrap + u64::from(raw_t);
            prev_t = t;
            let x = (data & 0x3FFF) as u16;
            let y = ((data >> 14) & 0x3FFF) as u16;
            if let Some(w) = width
                && w != 0
                && x >= w
            {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at + 4,
                    field: "column",
                    value: u64::from(x),
                    max: u64::from(w - 1),
                });
            }
            if let Some(h) = height
                && h != 0
                && y >= h
            {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at + 4,
                    field: "row",
                    value: u64::from(y),
                    max: u64::from(h - 1),
                });
            }
            let polarity = if (data >> 28) & 1 == 1 { Polarity::On } else { Polarity::Off };
            events.push(AerEvent { t, x, y, polarity });
            at += 8;
        }
        Ok(Self { header, record_type, record_size, width, height, events })
    }

    /// Encode events as a `.dat` file.
    ///
    /// `record_type` is written through unchanged so that a decode-then-encode returns the file's
    /// own descriptor; pass `Dat::CD_TYPE_CODES[0]` for a fresh file.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are out of order;
    /// [`EncodeError::FieldOutOfRange`] if a coordinate exceeds [`Dat::MAX_COORD`] or a `Width` or
    /// `Height` line in `header` that the caller supplied;
    /// [`EncodeError::HeaderLineContainsNewline`] if a header line would not survive the round trip.
    pub fn encode(
        events: &[AerEvent],
        header: &[String],
        record_type: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        check_sorted(events)?;
        let mut out = Vec::with_capacity(64 + events.len() * 8);
        push_header_lines(&mut out, header, b'%', false)?;
        out.push(record_type);
        out.push(Self::RECORD_SIZE);
        let w_lim = Self::header_number(header, "Width")
            .map_or(Self::MAX_COORD, |w| Self::MAX_COORD.min(u64::from(w.saturating_sub(1))));
        let h_lim = Self::header_number(header, "Height")
            .map_or(Self::MAX_COORD, |h| Self::MAX_COORD.min(u64::from(h.saturating_sub(1))));
        for (i, e) in events.iter().enumerate() {
            fit(i, "column", u64::from(e.x), w_lim)?;
            fit(i, "row", u64::from(e.y), h_lim)?;
            let ts = (e.t & 0xFFFF_FFFF) as u32;
            let p = u32::from(e.polarity == Polarity::On);
            let data = u32::from(e.x) | (u32::from(e.y) << 14) | (p << 28);
            out.extend_from_slice(&ts.to_le_bytes());
            out.extend_from_slice(&data.to_le_bytes());
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Flat — this crate's own interchange
// ---------------------------------------------------------------------------------------------

/// This crate's own flat interchange format: fixed 16-byte records, no state, no opcode.
///
/// It exists because the five vendor formats above each have a reason to be awkward — a 24-bit
/// clock, a chip-specific bit layout, a compressor — and a teaching exercise, a test fixture or a
/// cross-language hand-off needs none of that. The whole specification is:
///
/// ```text
/// bytes  0- 7  magic "FMAER-01"
/// bytes  8-11  uint32 width,  little-endian (0 = unstated)
/// bytes 12-15  uint32 height, little-endian (0 = unstated)
/// bytes 16-23  uint64 event count, little-endian
/// then `count` records of 16 bytes:
///   bytes 0- 7  uint64 timestamp, microseconds
///   bytes 8- 9  uint16 column
///   bytes 10-11 uint16 row
///   byte  12    polarity, 0 = Off, 1 = On
///   bytes 13-15 reserved, MUST be zero
/// ```
///
/// Little-endian throughout, 64 bits of time so nothing wraps within any recording a machine can
/// store, and a declared count so a truncated file is detected at the header rather than at the
/// last record. The three reserved bytes are **checked** to be zero from this first version, so
/// that a later version can use them; reserved bytes that are allowed to hold rubbish can never be
/// reclaimed.
///
/// The cost is honest and stated: 16 bytes per event is 4x `EVT` 2.0 and up to 12x a vectorised
/// `EVT` 3.0. This is an interchange and a fixture format, not a recording format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flat {
    /// Sensor width in pixels, or 0 if the writer did not state one.
    pub width: u16,
    /// Sensor height in pixels, or 0 if the writer did not state one.
    pub height: u16,
    /// Events in file order.
    pub events: Vec<AerEvent>,
}

impl Flat {
    /// The eight magic bytes every file starts with.
    pub const MAGIC: [u8; 8] = *b"FMAER-01";
    /// Bytes per record.
    pub const RECORD_SIZE: usize = 16;
    /// Bytes of fixed header before the first record.
    pub const HEADER_SIZE: usize = 24;

    /// Decode a flat file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::BadMagic`] if the first eight bytes are not [`Flat::MAGIC`];
    /// [`DecodeError::Truncated`] if the header is short; [`DecodeError::CountMismatch`] if the
    /// declared count and the bytes present disagree — in either direction, because trailing bytes
    /// after the last record mean the file is not what its header says it is;
    /// [`DecodeError::FieldOutOfRange`] for a polarity byte that is neither 0 nor 1, or a
    /// coordinate past a declared width or height; [`DecodeError::ReservedNotZero`] for a dirty
    /// reserved byte; [`DecodeError::NonMonotonicTimestamp`] if the events are out of order.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let head = slice_at(bytes, 0, Self::HEADER_SIZE)?;
        if head[..8] != Self::MAGIC {
            return Err(DecodeError::BadMagic {
                offset: 0,
                expected: "FMAER-01",
                found: String::from_utf8_lossy(&head[..8]).into_owned(),
            });
        }
        let width = u16::try_from(u32_le(bytes, 8)?).unwrap_or(u16::MAX);
        let height = u16::try_from(u32_le(bytes, 12)?).unwrap_or(u16::MAX);
        let declared = u64_le(bytes, 16)?;
        let body = bytes.len() - Self::HEADER_SIZE;
        let actual = (body / Self::RECORD_SIZE) as u64;
        if declared != actual || !body.is_multiple_of(Self::RECORD_SIZE) {
            return Err(DecodeError::CountMismatch { offset: 16, declared, actual });
        }
        let mut events = Vec::with_capacity(actual as usize);
        let mut last_t = 0u64;
        for k in 0..actual as usize {
            let at = Self::HEADER_SIZE + k * Self::RECORD_SIZE;
            let t = u64_le(bytes, at)?;
            if t < last_t {
                return Err(DecodeError::NonMonotonicTimestamp {
                    offset: at,
                    previous: last_t,
                    found: t,
                });
            }
            last_t = t;
            let x = u16_le(bytes, at + 8)?;
            let y = u16_le(bytes, at + 10)?;
            let rec = slice_at(bytes, at + 12, 4)?;
            let polarity = match rec[0] {
                0 => Polarity::Off,
                1 => Polarity::On,
                other => {
                    return Err(DecodeError::FieldOutOfRange {
                        offset: at + 12,
                        field: "polarity",
                        value: u64::from(other),
                        max: 1,
                    });
                }
            };
            for (j, &b) in rec[1..].iter().enumerate() {
                if b != 0 {
                    return Err(DecodeError::ReservedNotZero { offset: at + 13 + j });
                }
            }
            if width != 0 && x >= width {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at + 8,
                    field: "column",
                    value: u64::from(x),
                    max: u64::from(width - 1),
                });
            }
            if height != 0 && y >= height {
                return Err(DecodeError::FieldOutOfRange {
                    offset: at + 10,
                    field: "row",
                    value: u64::from(y),
                    max: u64::from(height - 1),
                });
            }
            events.push(AerEvent { t, x, y, polarity });
        }
        Ok(Self { width, height, events })
    }

    /// Encode events as a flat file. Pass `0` for a width or height that is not known.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order, and
    /// [`EncodeError::FieldOutOfRange`] if a coordinate is at or past a non-zero `width` or
    /// `height` — a stated geometry that the events contradict is a bug worth stopping for.
    pub fn encode(events: &[AerEvent], width: u16, height: u16) -> Result<Vec<u8>, EncodeError> {
        check_sorted(events)?;
        let mut out = Vec::with_capacity(Self::HEADER_SIZE + events.len() * Self::RECORD_SIZE);
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&u32::from(width).to_le_bytes());
        out.extend_from_slice(&u32::from(height).to_le_bytes());
        out.extend_from_slice(&(events.len() as u64).to_le_bytes());
        for (i, e) in events.iter().enumerate() {
            if width != 0 {
                fit(i, "column", u64::from(e.x), u64::from(width) - 1)?;
            }
            if height != 0 {
                fit(i, "row", u64::from(e.y), u64::from(height) - 1)?;
            }
            out.extend_from_slice(&e.t.to_le_bytes());
            out.extend_from_slice(&e.x.to_le_bytes());
            out.extend_from_slice(&e.y.to_le_bytes());
            out.push(u8::from(e.polarity == Polarity::On));
            out.extend_from_slice(&[0u8; 3]);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// AEDAT 4.0 — framing in full, payloads as far as a zero-dependency crate can go
// ---------------------------------------------------------------------------------------------

/// How an `AEDAT` 4.0 packet payload is compressed, as the file's `Format:` header line states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Aedat4Compression {
    /// `RAW`: the payload is an uncompressed `FlatBuffers` table. This is the only variant whose
    /// events this crate can extract.
    Raw,
    /// `COMPRESSED_LZ4`.
    Lz4,
    /// `COMPRESSED_LZ4_HIGH`.
    Lz4High,
    /// `COMPRESSED_ZSTD`.
    Zstd,
    /// `COMPRESSED_ZSTD_HIGH`.
    ZstdHigh,
    /// A `Format:` value this implementation did not recognise, kept verbatim so an error can name
    /// it and a future reader can match on it.
    Other(String),
}

impl Aedat4Compression {
    /// The `Format:` string this variant corresponds to.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Raw => "RAW",
            Self::Lz4 => "COMPRESSED_LZ4",
            Self::Lz4High => "COMPRESSED_LZ4_HIGH",
            Self::Zstd => "COMPRESSED_ZSTD",
            Self::ZstdHigh => "COMPRESSED_ZSTD_HIGH",
            Self::Other(s) => s,
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "RAW" => Self::Raw,
            "COMPRESSED_LZ4" => Self::Lz4,
            "COMPRESSED_LZ4_HIGH" => Self::Lz4High,
            "COMPRESSED_ZSTD" => Self::Zstd,
            "COMPRESSED_ZSTD_HIGH" => Self::ZstdHigh,
            other => Self::Other(other.to_string()),
        }
    }
}

/// One `AEDAT` 4.0 packet: an 8-byte header and a payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat4Packet {
    /// The stream this packet belongs to, matching an entry in the file's `IOHeader`. A recording
    /// interleaves streams — events, frames, IMU — and the id is the only thing that separates them.
    pub stream_id: i32,
    /// Byte offset of this packet's 8-byte header in the whole file.
    pub offset: usize,
    /// The payload bytes exactly as they appeared, kept so that a re-encode of a packet this crate
    /// could not decode is byte-identical rather than lost.
    pub payload: Vec<u8>,
    /// Events, when the payload was `RAW` and parsed; `None` when it was compressed or was not an
    /// event packet this implementation could walk.
    pub events: Option<Vec<AerEvent>>,
}

/// A decoded `AEDAT` 4.0 file — **framing complete, payload decoding partial, and this doc says
/// exactly where the line is**.
///
/// The container is: `#`-prefixed `\r\n` header lines ending with `#!END-HEADER`, a 4-byte
/// little-endian size followed by that many bytes of `IOHeader`, and then a sequence of packets,
/// each an 8-byte header — little-endian `int32` stream id, little-endian `int32` payload size —
/// followed by the payload. [`Aedat4::decode`] reads all of that, for every file, whatever the
/// compression: you always get the stream ids, the packet boundaries and the byte offsets.
///
/// # What this implementation cannot do
///
/// **Compressed payloads.** `DV` writes `LZ4` or `Zstd` by default. This crate has zero
/// dependencies and carries no decompressor, so a compressed packet arrives with `events: None` and
/// [`Aedat4::events`] returns [`DecodeError::UnsupportedCompression`] naming the format and the
/// offset. That is a capability boundary of a zero-dependency crate, stated rather than hidden;
/// re-save the recording as `RAW` from `DV`, or decompress upstream.
///
/// **Certainty about the `FlatBuffers` layout.** For a `RAW` payload this implementation walks the
/// buffer by hand — root offset, vtable, then a vector of 16-byte inline structs laid out as
/// `int64` timestamp, `int16` column, `int16` row, `bool` polarity, three bytes of padding — which
/// is the `events.fbs` schema of `dv-processing` as this implementation reads it. **It was not
/// checked against a file written by `DV` software**, because this review did not locate a `DV`
/// recording it could verify byte for byte. What *is* checked is that the reader and the writer
/// here agree exactly over thousands of events, and that the reader rejects every malformed buffer
/// it is given rather than reading past the end. Treat this codec as production-ready for the
/// framing and as unverified for the payload, and say so if you publish a figure that depends on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat4 {
    /// Header lines with their leading `#` and trailing `\r\n` removed, `!END-HEADER` included.
    pub header: Vec<String>,
    /// The compression the `Format:` header line declared, defaulting to `RAW` if it said nothing.
    pub compression: Aedat4Compression,
    /// The `IOHeader` blob verbatim. It is a `FlatBuffers` table describing the streams; this
    /// implementation preserves it byte for byte and does not interpret it, so a re-encode is
    /// lossless and a caller with a `FlatBuffers` reader can parse it separately.
    pub io_header: Vec<u8>,
    /// Packets in file order.
    pub packets: Vec<Aedat4Packet>,
}

impl Aedat4 {
    /// The magic the first header line must start with.
    pub const MAGIC: &'static str = "!AER-DAT4";
    /// The header line that terminates the ASCII header.
    pub const END_HEADER: &'static str = "!END-HEADER";
    /// Bytes per event inside a `RAW` payload's `FlatBuffers` struct vector.
    pub const FB_EVENT_SIZE: usize = 16;

    /// Decode the container, and the payloads that are `RAW`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::BadMagic`] if the first line is not `#!AER-DAT4...`;
    /// [`DecodeError::Truncated`] if the header has no `#!END-HEADER`, or a size field or payload
    /// runs off the end; [`DecodeError::FieldOutOfRange`] for a negative size field or a negative
    /// coordinate or timestamp inside a payload; [`DecodeError::MalformedFlatBuffer`] if a `RAW`
    /// payload's offsets are inconsistent; [`DecodeError::NonMonotonicTimestamp`] if a payload's
    /// events go backwards.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        // A dedicated header loop rather than `read_header_lines`: the byte after the header is
        // binary, and 0x23 is both the first byte of a small size field and the character '#'.
        // Stopping at `#!END-HEADER` is what keeps a size field of 35 from being eaten as a
        // comment line.
        let mut header: Vec<String> = Vec::new();
        let mut at = 0usize;
        let mut ended = false;
        while !ended {
            if bytes.get(at) != Some(&b'#') {
                // At offset zero this is the wrong file, not a short one, and saying so is what
                // stops another format's first bytes being walked as a header.
                return if at == 0 {
                    Err(DecodeError::BadMagic {
                        offset: 0,
                        expected: "#!AER-DAT4.0",
                        found: String::from_utf8_lossy(&bytes[..bytes.len().min(12)]).into_owned(),
                    })
                } else {
                    Err(DecodeError::Truncated {
                        offset: at,
                        need: Self::END_HEADER.len() + 3,
                        have: bytes.len().saturating_sub(at),
                    })
                };
            }
            let start = at + 1;
            let nl = bytes[start..].iter().position(|&c| c == b'\n').ok_or(
                DecodeError::Truncated {
                    offset: at,
                    need: bytes.len() - at + 1,
                    have: bytes.len() - at,
                },
            )?;
            let end = start + nl;
            let text_end = if end > start && bytes[end - 1] == b'\r' { end - 1 } else { end };
            let line = String::from_utf8_lossy(&bytes[start..text_end]).into_owned();
            if header.is_empty() && !line.starts_with(Self::MAGIC) {
                return Err(DecodeError::BadMagic {
                    offset: 0,
                    expected: "#!AER-DAT4.0",
                    found: String::from_utf8_lossy(&bytes[..bytes.len().min(12)]).into_owned(),
                });
            }
            ended = line == Self::END_HEADER;
            header.push(line);
            at = end + 1;
        }

        let compression = header
            .iter()
            .find_map(|l| l.strip_prefix("Format:"))
            .map_or(Aedat4Compression::Raw, |v| Aedat4Compression::parse(v.trim()));

        let io_len = Self::signed_len(bytes, at, "IOHeader size")?;
        at += 4;
        let io_header = slice_at(bytes, at, io_len)?.to_vec();
        at += io_len;

        let mut packets = Vec::new();
        // Carried ACROSS packets: a file whose packet boundaries each hold sorted events but whose
        // packets are out of order would otherwise pass every per-packet check and still hand the
        // caller a non-monotonic stream. A mutation fuzzer found exactly that.
        let mut last_t = 0u64;
        while at < bytes.len() {
            let head = at;
            let stream_id = u32_le(bytes, at)? as i32;
            let size = Self::signed_len(bytes, at + 4, "packet size")?;
            let payload = slice_at(bytes, at + 8, size)?.to_vec();
            let events = if compression == Aedat4Compression::Raw {
                Some(Self::read_event_packet(&payload, at + 8, &mut last_t)?)
            } else {
                None
            };
            packets.push(Aedat4Packet { stream_id, offset: head, payload, events });
            at += 8 + size;
        }
        Ok(Self { header, compression, io_header, packets })
    }

    fn signed_len(bytes: &[u8], at: usize, field: &'static str) -> Result<usize, DecodeError> {
        let v = u32_le(bytes, at)? as i32;
        usize::try_from(v).map_err(|_| DecodeError::FieldOutOfRange {
            offset: at,
            field,
            value: u64::from(v.unsigned_abs()),
            max: u64::from(i32::MAX.unsigned_abs()),
        })
    }

    /// Walk a `RAW` `FlatBuffers` event packet. Every offset is bounds-checked before use.
    fn read_event_packet(
        p: &[u8],
        base: usize,
        last_t: &mut u64,
    ) -> Result<Vec<AerEvent>, DecodeError> {
        let bad = |what: &'static str| DecodeError::MalformedFlatBuffer { offset: base, what };
        if p.len() < 8 {
            return Err(bad("root offset"));
        }
        let root = u32_le(p, 0).map_err(|_| bad("root offset"))? as usize;
        if root + 4 > p.len() {
            return Err(bad("root offset"));
        }
        let soffset = u32_le(p, root).map_err(|_| bad("vtable"))? as i32;
        let vtable = i64::try_from(root).map_err(|_| bad("vtable"))? - i64::from(soffset);
        let vtable = usize::try_from(vtable).map_err(|_| bad("vtable"))?;
        if vtable + 4 > p.len() {
            return Err(bad("vtable"));
        }
        let vt_len = u16_le(p, vtable).map_err(|_| bad("vtable"))? as usize;
        if vt_len < 6 || vtable + vt_len > p.len() {
            // A vtable shorter than 6 has no fields at all, so there is no element vector to find.
            return if vt_len < 6 && vtable + vt_len <= p.len() {
                Ok(Vec::new())
            } else {
                Err(bad("vtable"))
            };
        }
        let field = u16_le(p, vtable + 4).map_err(|_| bad("vtable"))? as usize;
        if field == 0 {
            return Ok(Vec::new());
        }
        let slot = root + field;
        if slot + 4 > p.len() {
            return Err(bad("vector"));
        }
        let vec_at = slot + u32_le(p, slot).map_err(|_| bad("vector"))? as usize;
        if vec_at + 4 > p.len() {
            return Err(bad("vector"));
        }
        let count = u32_le(p, vec_at).map_err(|_| bad("vector"))? as usize;
        let bytes_needed = count.checked_mul(Self::FB_EVENT_SIZE).ok_or_else(|| bad("vector"))?;
        if vec_at + 4 + bytes_needed > p.len() {
            return Err(DecodeError::CountMismatch {
                offset: base + vec_at,
                declared: count as u64,
                actual: ((p.len() - vec_at - 4) / Self::FB_EVENT_SIZE) as u64,
            });
        }
        let mut events = Vec::with_capacity(count);
        for k in 0..count {
            let at = vec_at + 4 + k * Self::FB_EVENT_SIZE;
            let raw_t = u64_le(p, at).map_err(|_| bad("vector"))? as i64;
            let t = u64::try_from(raw_t).map_err(|_| DecodeError::FieldOutOfRange {
                offset: base + at,
                field: "timestamp",
                value: raw_t.unsigned_abs(),
                max: u64::try_from(i64::MAX).unwrap_or(u64::MAX),
            })?;
            if t < *last_t {
                return Err(DecodeError::NonMonotonicTimestamp {
                    offset: base + at,
                    previous: *last_t,
                    found: t,
                });
            }
            *last_t = t;
            let sx = u16_le(p, at + 8).map_err(|_| bad("vector"))? as i16;
            let sy = u16_le(p, at + 10).map_err(|_| bad("vector"))? as i16;
            for (v, field) in [(sx, "column"), (sy, "row")] {
                if v < 0 {
                    return Err(DecodeError::FieldOutOfRange {
                        offset: base + at + 8,
                        field,
                        value: u64::from(v.unsigned_abs()),
                        max: u64::from(i16::MAX.unsigned_abs()),
                    });
                }
            }
            let pol_byte = *p.get(at + 12).ok_or_else(|| bad("vector"))?;
            let polarity = match pol_byte {
                0 => Polarity::Off,
                1 => Polarity::On,
                other => {
                    return Err(DecodeError::FieldOutOfRange {
                        offset: base + at + 12,
                        field: "polarity",
                        value: u64::from(other),
                        max: 1,
                    });
                }
            };
            events.push(AerEvent { t, x: sx.unsigned_abs(), y: sy.unsigned_abs(), polarity });
        }
        Ok(events)
    }

    /// All events from every packet, in file order.
    ///
    /// # Errors
    ///
    /// [`DecodeError::UnsupportedCompression`] naming the packet's byte offset and the file's
    /// declared format, if any packet's payload was not decoded. This is where the crate's
    /// zero-dependency boundary becomes visible to a caller, and it refuses with the reason rather
    /// than returning a partial list that looks like a short recording.
    pub fn events(&self) -> Result<Vec<AerEvent>, DecodeError> {
        let mut all = Vec::new();
        for p in &self.packets {
            match &p.events {
                Some(e) => all.extend_from_slice(e),
                None => {
                    return Err(DecodeError::UnsupportedCompression {
                        offset: p.offset,
                        name: self.compression.name().to_string(),
                    });
                }
            }
        }
        Ok(all)
    }

    /// Build a `RAW` `AEDAT` 4.0 file from events, one packet per `events_per_packet`.
    ///
    /// The `IOHeader` is written as an empty blob: this implementation does not synthesise a
    /// stream description it cannot verify against `DV`. A file produced here therefore round-trips
    /// through [`Aedat4::decode`] exactly and should be assumed **not** to open in `DV` until
    /// someone checks it.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order, or
    /// [`EncodeError::FieldOutOfRange`] if a coordinate exceeds 32767, the signed 16-bit field the
    /// schema uses.
    pub fn raw_from_events(
        events: &[AerEvent],
        events_per_packet: usize,
        stream_id: i32,
    ) -> Result<Self, EncodeError> {
        check_sorted(events)?;
        let chunk = events_per_packet.max(1);
        let mut packets = Vec::new();
        for (c, part) in events.chunks(chunk).enumerate() {
            for (k, e) in part.iter().enumerate() {
                let i = c * chunk + k;
                fit(i, "column", u64::from(e.x), 32767)?;
                fit(i, "row", u64::from(e.y), 32767)?;
                fit(i, "timestamp", e.t, u64::try_from(i64::MAX).unwrap_or(u64::MAX))?;
            }
            packets.push(Aedat4Packet {
                stream_id,
                offset: 0,
                payload: Self::write_event_packet(part),
                events: Some(part.to_vec()),
            });
        }
        Ok(Self {
            header: vec![
                "!AER-DAT4.0".to_string(),
                "Format: RAW".to_string(),
                Self::END_HEADER.to_string(),
            ],
            compression: Aedat4Compression::Raw,
            io_header: Vec::new(),
            packets,
        })
    }

    /// The canonical `FlatBuffers` table this implementation writes and reads.
    ///
    /// Laid out forwards, which a general `FlatBuffers` writer does not do — it builds backwards —
    /// but which makes the byte offsets checkable by eye against the reader above:
    ///
    /// ```text
    ///  0.. 4  uint32 root offset = 12
    ///  4.. 6  uint16 vtable length = 6
    ///  6.. 8  uint16 table length = 8
    ///  8..10  uint16 offset of field 0 within the table = 4
    /// 10..12  padding, aligning the table to 4
    /// 12..16  int32 soffset to the vtable = 8
    /// 16..20  uint32 offset from this slot to the vector = 4
    /// 20..24  uint32 element count
    /// 24..    elements, 16 bytes each, 8-aligned as the int64 field requires
    /// ```
    fn write_event_packet(events: &[AerEvent]) -> Vec<u8> {
        let mut p = Vec::with_capacity(24 + events.len() * Self::FB_EVENT_SIZE);
        p.extend_from_slice(&12u32.to_le_bytes());
        p.extend_from_slice(&6u16.to_le_bytes());
        p.extend_from_slice(&8u16.to_le_bytes());
        p.extend_from_slice(&4u16.to_le_bytes());
        p.extend_from_slice(&0u16.to_le_bytes());
        p.extend_from_slice(&8i32.to_le_bytes());
        p.extend_from_slice(&4u32.to_le_bytes());
        p.extend_from_slice(&(events.len() as u32).to_le_bytes());
        for e in events {
            p.extend_from_slice(&(e.t as i64).to_le_bytes());
            p.extend_from_slice(&(e.x as i16).to_le_bytes());
            p.extend_from_slice(&(e.y as i16).to_le_bytes());
            p.push(u8::from(e.polarity == Polarity::On));
            p.extend_from_slice(&[0u8; 3]);
        }
        p
    }

    /// Serialise back to bytes.
    ///
    /// A packet that was decoded is re-serialised from its events; a packet that was not — because
    /// it was compressed — is written from its preserved payload, byte for byte. So a compressed
    /// file survives a decode-encode cycle unchanged even though this crate cannot read inside it.
    ///
    /// # Errors
    ///
    /// [`EncodeError::HeaderLineContainsNewline`] if a header line would not survive the round
    /// trip. Field ranges were already enforced when the packets were built.
    pub fn encode(&self) -> Result<Vec<u8>, EncodeError> {
        let mut out = Vec::new();
        push_header_lines(&mut out, &self.header, b'#', true)?;
        let io_len = i32::try_from(self.io_header.len()).unwrap_or(i32::MAX);
        out.extend_from_slice(&io_len.to_le_bytes());
        out.extend_from_slice(&self.io_header);
        for p in &self.packets {
            let payload = match (&p.events, &self.compression) {
                (Some(e), Aedat4Compression::Raw) => Self::write_event_packet(e),
                _ => p.payload.clone(),
            };
            let size = i32::try_from(payload.len()).unwrap_or(i32::MAX);
            out.extend_from_slice(&p.stream_id.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&payload);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// Into the rest of the crate
// ---------------------------------------------------------------------------------------------

/// Turn decoded sensor events into a [`crate::spike::Train`] this crate's simulator can consume.
///
/// Three decisions have to be made at this boundary, and making them explicit fields is the point
/// of the struct: how long a tick is, how the 2-D array flattens to a 1-D address, and what happens
/// to the polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainMap {
    /// Sensor width in pixels. Addresses are `y * width + x`.
    pub width: u16,
    /// Sensor height in pixels, needed only when [`TrainMap::split_polarity`] is set.
    pub height: u16,
    /// Microseconds per simulator tick. A sensor timestamp is divided by this, **rounding down**,
    /// so several events can land on one tick — which is what a synchronous fabric genuinely does
    /// to them, and is why the crate's simulator counts a tick's arrivals rather than assuming one.
    pub tick_us: u64,
    /// Whether `On` and `Off` become two separate address planes, `Off` first, `On` offset by
    /// `width * height`.
    ///
    /// The alternative — one plane, sign discarded — halves the address space and throws away the
    /// contrast direction while leaving the event rate unchanged, which is invisible in a raster
    /// plot. Set this unless you have a reason, and if you do not set it, say so in the caption.
    pub split_polarity: bool,
}

impl TrainMap {
    /// Map events to a spike train.
    ///
    /// Returns `None` if `tick_us` is zero, if any event lies outside the stated geometry, or if an
    /// address would not fit a `u32`. A refusal rather than a clamp: an out-of-range coordinate
    /// means this map does not describe the sensor that produced the file.
    #[must_use]
    pub fn map(&self, events: &[AerEvent]) -> Option<Train> {
        if self.tick_us == 0 || self.width == 0 || self.height == 0 {
            return None;
        }
        let plane = u32::from(self.width).checked_mul(u32::from(self.height))?;
        let mut spikes = Vec::with_capacity(events.len());
        for e in events {
            if e.x >= self.width || e.y >= self.height {
                return None;
            }
            let mut address =
                u32::from(e.y).checked_mul(u32::from(self.width))?.checked_add(u32::from(e.x))?;
            if self.split_polarity && e.polarity == Polarity::On {
                address = address.checked_add(plane)?;
            }
            spikes.push(crate::spike::Spike { t: e.t / self.tick_us, source: address });
        }
        Some(Train::from_spikes(spikes))
    }

    /// How many distinct addresses a network consuming this map needs.
    ///
    /// `width * height`, doubled when polarity is split. This is the neuron count for an input
    /// layer, and getting it wrong by the factor of two is the most common way an event-driven
    /// pipeline silently drops every `On` event.
    #[must_use]
    pub fn address_count(&self) -> Option<u32> {
        let plane = u32::from(self.width).checked_mul(u32::from(self.height))?;
        if self.split_polarity { plane.checked_mul(2) } else { Some(plane) }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Aedat2, Aedat2Layout, Aedat4, Aedat4Compression, AerEvent, Dat, DecodeError, EncodeError,
        Evt2, Evt3, Flat, MarkerKind, TrainMap,
    };
    use crate::rng::Rng;
    use crate::spike::Polarity;

    /// A deterministic synthetic stream: `n` events sorted in time, coordinates filling the whole
    /// `w` by `h` array, both polarities, timestamps advancing by 0 to `max_step` microseconds so
    /// that same-microsecond bursts and long gaps both occur.
    fn synth(n: usize, seed: u64, w: u16, h: u16, max_step: u32) -> Vec<AerEvent> {
        let mut r = Rng::new(seed);
        let mut t = 0u64;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            t += u64::from(r.below(max_step + 1));
            v.push(AerEvent {
                t,
                x: r.below(u32::from(w)) as u16,
                y: r.below(u32::from(h)) as u16,
                polarity: if r.next_u32() & 1 == 0 { Polarity::Off } else { Polarity::On },
            });
        }
        v
    }

    fn monotonic(events: &[AerEvent]) -> bool {
        events.windows(2).all(|p| p[0].t <= p[1].t)
    }

    // -----------------------------------------------------------------------------------------
    // (a) Encode-decode round trips, bit-exact, thousands of events, full ranges.
    // -----------------------------------------------------------------------------------------

    /// `AEDAT` 2.0 for both preset layouts. The `DAVIS346` case matters because its fields are not
    /// byte-aligned and its row field is 9 bits, so an off-by-one in a mask shows up as a row that
    /// is folded rather than as a failure to decode.
    #[test]
    fn aedat2_round_trips_bit_exactly_for_both_layouts() {
        for layout in [Aedat2Layout::DVS128, Aedat2Layout::DAVIS346] {
            let events = synth(4_000, 0xA1, layout.width, layout.height, 400);
            let header = vec!["!AER-DAT2.0".to_string(), " synthetic, ferromorphic".to_string()];
            let bytes = Aedat2::encode(&events, layout, &header).expect("encodable");
            let back = Aedat2::decode(&bytes, layout).expect("decodable");
            assert_eq!(back.events, events, "{}", layout.source);
            assert_eq!(back.header, header);
            assert!(back.markers.is_empty());
            // 8 bytes per record plus the header, exactly. A format that spent a byte more would
            // still round trip; this is the claim that the record is the one the spec describes.
            let header_bytes: usize = header.iter().map(|l| l.len() + 3).sum();
            assert_eq!(bytes.len(), header_bytes + 8 * events.len());
        }
    }

    #[test]
    fn evt2_round_trips_bit_exactly() {
        let events = synth(4_000, 0xB2, 1280, 720, 300);
        let bytes = Evt2::encode(&events).expect("encodable");
        let back = Evt2::decode(&bytes).expect("decodable");
        assert_eq!(back.events, events);
        assert!(monotonic(&back.events));
    }

    #[test]
    fn evt3_round_trips_bit_exactly_in_both_encodings() {
        let events = synth(4_000, 0xC3, 1280, 720, 300);
        for vectorise in [false, true] {
            let bytes = Evt3::encode(&events, vectorise).expect("encodable");
            let back = Evt3::decode(&bytes).expect("decodable");
            assert_eq!(back.events, events, "vectorise = {vectorise}");
            assert!(monotonic(&back.events));
        }
    }

    #[test]
    fn dat_round_trips_bit_exactly() {
        let header =
            vec![" Data file containing CD events.".to_string(), " Width 1280".to_string(), " Height 720".to_string()];
        let events = synth(4_000, 0xD4, 1280, 720, 300);
        for &code in &Dat::CD_TYPE_CODES {
            let bytes = Dat::encode(&events, &header, code).expect("encodable");
            let back = Dat::decode(&bytes).expect("decodable");
            assert_eq!(back.events, events);
            assert_eq!(back.record_type, code, "the file's own type byte is preserved");
            assert_eq!(back.width, Some(1280));
            assert_eq!(back.height, Some(720));
        }
    }

    #[test]
    fn flat_round_trips_bit_exactly() {
        let events = synth(4_000, 0xE5, 640, 480, 500);
        let bytes = Flat::encode(&events, 640, 480).expect("encodable");
        assert_eq!(bytes.len(), Flat::HEADER_SIZE + Flat::RECORD_SIZE * events.len());
        let back = Flat::decode(&bytes).expect("decodable");
        assert_eq!(back.events, events);
        assert_eq!((back.width, back.height), (640, 480));
    }

    #[test]
    fn aedat4_raw_round_trips_bit_exactly_through_several_packets() {
        let events = synth(4_000, 0xF6, 346, 260, 200);
        let file = Aedat4::raw_from_events(&events, 512, 7).expect("encodable");
        assert_eq!(file.packets.len(), 8, "4000 events in packets of 512");
        let bytes = file.encode().expect("serialisable");
        let back = Aedat4::decode(&bytes).expect("decodable");
        assert_eq!(back.compression, Aedat4Compression::Raw);
        assert_eq!(back.events().expect("all RAW"), events);
        assert!(back.packets.iter().all(|p| p.stream_id == 7));
        // And the container itself is byte-stable, which is what lets a pipeline re-save a file
        // it only partly understands.
        assert_eq!(back.encode().expect("serialisable"), bytes);
    }

    /// Every format decodes to the same events from the same input. This is the check that the six
    /// codecs share one semantics rather than six nearly-identical ones.
    #[test]
    fn all_six_formats_agree_on_the_same_events() {
        let events = synth(2_000, 0x17, 640, 480, 250);
        let header = vec![" Width 640".to_string(), " Height 480".to_string()];
        let via_flat = Flat::decode(&Flat::encode(&events, 640, 480).unwrap()).unwrap().events;
        let via_evt2 = Evt2::decode(&Evt2::encode(&events).unwrap()).unwrap().events;
        let via_evt3 = Evt3::decode(&Evt3::encode(&events, true).unwrap()).unwrap().events;
        let via_dat = Dat::decode(&Dat::encode(&events, &header, 0x00).unwrap()).unwrap().events;
        let via_a2 = Aedat2::decode(
            &Aedat2::encode(&events, Aedat2Layout::DAVIS346, &[]).unwrap_or_default(),
            Aedat2Layout::DAVIS346,
        );
        let via_a4 = Aedat4::raw_from_events(&events, 1024, 0).unwrap().events().unwrap();
        assert_eq!(via_flat, events);
        assert_eq!(via_evt2, events);
        assert_eq!(via_evt3, events);
        assert_eq!(via_dat, events);
        assert_eq!(via_a4, events);
        // AEDAT 2.0 with the DAVIS346 layout cannot hold a 640x480 array, and says so rather than
        // folding the columns. That refusal is the point of asserting it here.
        assert!(via_a2.is_err() || Aedat2::encode(&events, Aedat2Layout::DAVIS346, &[]).is_err());
    }

    // -----------------------------------------------------------------------------------------
    // Bit layouts, checked against the published field tables by hand-computed words.
    // -----------------------------------------------------------------------------------------

    /// `EVT` 2.0: `[type:4][time low:6][column:11][row:11]`, little-endian.
    ///
    /// One event at t = 683 us, column 1000, row 500, rising. 683 = 10 * 64 + 43, so the time-high
    /// word carries 10 and the event carries 43. The event word is therefore
    /// `0x1 << 28 | 43 << 22 | 1000 << 11 | 500` = `0x1ADF41F4`, which is written little-endian.
    #[test]
    fn evt2_words_match_the_published_field_table() {
        let e = AerEvent { t: 683, x: 1000, y: 500, polarity: Polarity::On };
        let bytes = Evt2::encode(&[e]).expect("encodable");
        assert_eq!(bytes, vec![0x0A, 0x00, 0x00, 0x80, 0xF4, 0x41, 0xDF, 0x1A]);
        assert_eq!(Evt2::decode(&bytes).unwrap().events, vec![e]);
    }

    /// `EVT` 3.0: 16-bit words, opcode in bits 15-12.
    ///
    /// One event at t = 0x123456 us, column 1000 (0x3E8), row 500 (0x1F4), rising. The stream is
    /// `TIME_HIGH 0x123`, `TIME_LOW 0x456`, `ADDR_Y 0x1F4`, `ADDR_X 0x3E8` with the polarity bit
    /// 11 set: `0x8123, 0x6456, 0x01F4, 0x2BE8`, little-endian.
    #[test]
    fn evt3_words_match_the_published_field_table() {
        let e = AerEvent { t: 0x0012_3456, x: 1000, y: 500, polarity: Polarity::On };
        let bytes = Evt3::encode(&[e], false).expect("encodable");
        assert_eq!(bytes, vec![0x23, 0x81, 0x56, 0x64, 0xF4, 0x01, 0xE8, 0x2B]);
        assert_eq!(Evt3::decode(&bytes).unwrap().events, vec![e]);
    }

    /// `.dat`: `uint32` microseconds then `column | row << 14 | polarity << 28`, little-endian.
    ///
    /// t = 0x12345678, column 1000, row 500, rising gives a payload of
    /// `1000 | 500 << 14 | 1 << 28` = `0x107D03E8`.
    #[test]
    fn dat_records_match_the_published_field_table() {
        let e = AerEvent { t: 0x1234_5678, x: 1000, y: 500, polarity: Polarity::On };
        let bytes = Dat::encode(&[e], &[], 0x00).expect("encodable");
        assert_eq!(bytes, vec![0x00, 0x08, 0x78, 0x56, 0x34, 0x12, 0xE8, 0x03, 0x7D, 0x10]);
        assert_eq!(Dat::decode(&bytes).unwrap().events, vec![e]);
    }

    /// `AEDAT` 2.0, `DVS128`: row in bits 8-14, column in bits 1-7 counted from the right edge,
    /// polarity in bit 0, whole record big-endian.
    ///
    /// Column 10 becomes 127 - 10 = 117 at bits 1-7, i.e. 234; row 20 becomes 20 << 8 = 5120; so
    /// the address is 5354 = `0x14EA`, and the record is `00 00 14 EA` then the timestamp, both
    /// big-endian — the only big-endian format in this module.
    #[test]
    fn aedat2_records_match_the_jaer_field_table() {
        let e = AerEvent { t: 0x1122_3344, x: 10, y: 20, polarity: Polarity::On };
        let bytes = Aedat2::encode(&[e], Aedat2Layout::DVS128, &[]).expect("encodable");
        assert_eq!(&bytes[bytes.len() - 8..], &[0x00, 0x00, 0x14, 0xEA, 0x11, 0x22, 0x33, 0x44]);
        assert_eq!(Aedat2::decode(&bytes, Aedat2Layout::DVS128).unwrap().events, vec![e]);
    }

    // -----------------------------------------------------------------------------------------
    // (b) EVT 3.0 timestamp reconstruction. The three tests the module doc names.
    // -----------------------------------------------------------------------------------------

    fn evt3_words(words: &[u16]) -> Vec<u8> {
        let mut v = Vec::with_capacity(words.len() * 2);
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    /// A hand-built stream crossing the 24-bit wrap three times.
    ///
    /// Every expected timestamp below is arithmetic done by hand from the field table, not a
    /// previous run of this decoder: 0xFFF << 12 = 16,773,120; one wrap adds 2^24 = 16,777,216.
    #[test]
    fn evt3_reconstructs_timestamps_across_repeated_rollovers() {
        let s = evt3_words(&[
            0x8000, 0x6001, 0x0005, 0x2000 | 10, // t = 1
            0x8FFF, 0x6FFF, 0x2000 | 11,         // t = 16,773,120 + 4,095 = 16,777,215
            0x8000, 0x6002, 0x2000 | 12,         // wrap 1: t = 16,777,216 + 2
            0x8FFF, 0x6FF0, 0x2000 | 13,         // t = 16,777,216 + 16,773,120 + 4,080
            0x8000, 0x6000, 0x2000 | 14,         // wrap 2: t = 33,554,432
        ]);
        let got = Evt3::decode(&s).expect("decodable");
        let times: Vec<u64> = got.events.iter().map(|e| e.t).collect();
        assert_eq!(times, vec![1, 16_777_215, 16_777_218, 33_554_416, 33_554_432]);
        assert!(monotonic(&got.events), "a missed wrap shows up here as a sawtooth");
        // And the columns rode along correctly, so this is not a test of time alone.
        assert_eq!(got.events.iter().map(|e| e.x).collect::<Vec<_>>(), vec![10, 11, 12, 13, 14]);
    }

    /// A low word smaller than its predecessor with no `TIME_HIGH` between them means the high word
    /// was dropped, and the decoder must advance the high itself.
    ///
    /// Here `TIME_HIGH 0x001` then `TIME_LOW 0x0FF` is 4096 + 255 = 4351; the next low is 0x00A,
    /// which is smaller, so the window advanced to 0x002 and the time is 8192 + 10 = 8202. A
    /// decoder without this branch reports 4106, which is EARLIER than the previous event.
    #[test]
    fn evt3_recovers_a_dropped_time_high_word() {
        let s = evt3_words(&[0x8001, 0x60FF, 0x0003, 0x2000 | 7, 0x600A, 0x2000 | 8]);
        let got = Evt3::decode(&s).expect("decodable");
        assert_eq!(got.events.iter().map(|e| e.t).collect::<Vec<_>>(), vec![4_351, 8_202]);
    }

    /// The mirror-image bug: a `TIME_HIGH` that *does* arrive must clear the remembered low, or the
    /// first small low in the new window is mistaken for a wrap and time jumps 4096 us.
    ///
    /// And a `TIME_HIGH` repeating the value already held must NOT clear it, or time rewinds inside
    /// its own window. Both branches are asserted here against hand arithmetic.
    #[test]
    fn evt3_does_not_invent_a_wrap_after_a_time_high() {
        let s = evt3_words(&[
            0x8001, 0x6FF0, 0x0001, 0x2000 | 1, // t = 4096 + 4080 = 8176
            0x8002, 0x6001, 0x2000 | 2,         // t = 8192 + 1 = 8193, NOT 12289
            0x8002, 0x2000 | 3,                 // repeated TIME_HIGH: still 8193, not 8192
        ]);
        let got = Evt3::decode(&s).expect("decodable");
        assert_eq!(got.events.iter().map(|e| e.t).collect::<Vec<_>>(), vec![8_176, 8_193, 8_193]);
    }

    /// A vector mask expands to exactly the columns whose bits are set, and the base advances by
    /// the mask width rather than by the number of events.
    ///
    /// `VECT_BASE_X 100` then `VECT_12 0xA05` — bits 0, 2, 9 and 11 — gives columns 100, 102, 109
    /// and 111. The base is then 112, and `VECT_8 0x81` — bits 0 and 7 — gives 112 and 119.
    #[test]
    fn evt3_vector_masks_expand_to_exactly_the_bits_set() {
        let s = evt3_words(&[
            0x8000,
            0x6000,
            0x0004,
            0x3000 | 0x800 | 100,
            0x4000 | 0xA05,
            0x5000 | 0x81,
        ]);
        let got = Evt3::decode(&s).expect("decodable");
        assert_eq!(
            got.events.iter().map(|e| e.x).collect::<Vec<_>>(),
            vec![100, 102, 109, 111, 112, 119]
        );
        assert!(got.events.iter().all(|e| e.y == 4 && e.t == 0 && e.polarity == Polarity::On));
    }

    /// The vector path must be a compression, not a second format: both encodings decode to the
    /// same events, and on a horizontally coherent stream the vectorised one is smaller.
    #[test]
    fn evt3_vectorised_and_plain_encodings_decode_identically() {
        // A synthetic horizontal edge: 64 adjacent columns firing in the same microsecond.
        let mut events = Vec::new();
        for frame in 0..40u64 {
            for x in 0..64u16 {
                events.push(AerEvent {
                    t: frame * 1_000,
                    x: 200 + x,
                    y: 100 + (frame as u16 % 7),
                    polarity: Polarity::On,
                });
            }
        }
        let plain = Evt3::encode(&events, false).expect("encodable");
        let vect = Evt3::encode(&events, true).expect("encodable");
        assert_eq!(Evt3::decode(&plain).unwrap().events, events);
        assert_eq!(Evt3::decode(&vect).unwrap().events, events);
        assert!(
            vect.len() * 4 < plain.len(),
            "vectorised {} bytes vs plain {} for a coherent edge",
            vect.len(),
            plain.len()
        );
    }

    // -----------------------------------------------------------------------------------------
    // (c) Totality. Truncation at every byte offset, and random and mutated input.
    // -----------------------------------------------------------------------------------------

    /// Six valid streams, one per format, small enough that every prefix of every one of them can
    /// be decoded in a test.
    fn corpus() -> Vec<(&'static str, Vec<u8>)> {
        let e = synth(120, 0x5EED, 128, 128, 200);
        let header = vec![" Width 128".to_string(), " Height 128".to_string()];
        vec![
            ("aedat2", Aedat2::encode(&e, Aedat2Layout::DVS128, &[]).unwrap()),
            ("evt2", Evt2::encode(&e).unwrap()),
            ("evt3-plain", Evt3::encode(&e, false).unwrap()),
            ("evt3-vect", Evt3::encode(&e, true).unwrap()),
            ("dat", Dat::encode(&e, &header, 0x0C).unwrap()),
            ("flat", Flat::encode(&e, 128, 128).unwrap()),
            ("aedat4", Aedat4::raw_from_events(&e, 32, 1).unwrap().encode().unwrap()),
        ]
    }

    /// What a sweep of malformed input actually exercised.
    ///
    /// Counted and asserted on, because a totality test is the easiest kind to make vacuous: a
    /// decoder that returned `Err` for everything would pass "it never panics" perfectly, and so
    /// would a test whose fixtures were all too short to reach the state machine.
    #[derive(Default, Debug)]
    struct Tally {
        ok: u64,
        err: u64,
        events: u64,
    }

    /// Decode with every decoder and assert the two properties that must hold for ANY input: the
    /// call returns, and if it returns events they are monotonically non-decreasing in time.
    ///
    /// Deliberately runs every decoder over every buffer, not just the matching one: reading a
    /// file with the wrong decoder is a routine accident and must fail rather than crash.
    fn decode_every_way(bytes: &[u8], tally: &mut Tally) {
        let note = |r: Result<Vec<AerEvent>, DecodeError>, tally: &mut Tally| match r {
            Ok(e) => {
                assert!(monotonic(&e));
                tally.ok += 1;
                tally.events += e.len() as u64;
            }
            Err(_) => tally.err += 1,
        };
        for layout in [Aedat2Layout::DVS128, Aedat2Layout::DAVIS346] {
            note(Aedat2::decode(bytes, layout).map(|f| f.events), tally);
        }
        note(Evt2::decode(bytes).map(|f| f.events), tally);
        note(Evt3::decode(bytes).map(|f| f.events), tally);
        note(Dat::decode(bytes).map(|f| f.events), tally);
        note(Flat::decode(bytes).map(|f| f.events), tally);
        note(Aedat4::decode(bytes).and_then(|f| f.events()), tally);
    }

    /// The test that makes this module safe to point at a file.
    ///
    /// Every prefix of every valid stream, for every decoder. A panic anywhere fails the test,
    /// because a panicking decoder is how a truncated recording — the normal result of a power
    /// loss or a cancelled copy — takes down whatever is reading it.
    #[test]
    fn every_truncation_of_every_format_errors_instead_of_panicking() {
        let mut tally = Tally::default();
        for (name, bytes) in corpus() {
            assert!(bytes.len() > 100, "{name} is too short to be a useful fixture");
            for cut in 0..=bytes.len() {
                decode_every_way(&bytes[..cut], &mut tally);
            }
        }
        // The sweep has to have gone both ways, or "it never panicked" says nothing.
        // Measured on this fixed seed: 1,341 successes, 57,025 refusals, 78,640 events decoded.
        // The bounds are set well below those so that an ordinary change does not trip them and a
        // decoder that stopped refusing, or stopped succeeding, does.
        assert!(tally.err > 5_000, "only {} refusals; is anything being refused?", tally.err);
        assert!(tally.ok > 500, "only {} successes; are the fixtures reaching the decoders?", tally.ok);
        assert!(tally.events > 10_000, "only {} events decoded in the whole sweep", tally.events);
    }

    /// Random bytes of random lengths, seeded so the failure is reproducible.
    #[test]
    fn random_bytes_never_panic() {
        let mut r = Rng::new(0x9E37_79B9);
        let mut tally = Tally::default();
        for _ in 0..4_000 {
            let n = r.below(600) as usize;
            let mut buf = Vec::with_capacity(n);
            for _ in 0..n {
                buf.push((r.next_u32() & 0xFF) as u8);
            }
            decode_every_way(&buf, &mut tally);
        }
        // Noise is almost always refused, and that is the point; the count is asserted so that a
        // future decoder which accepted noise would show up here rather than pass quietly.
        // Measured on this seed: 27,968 refusals against 32 successes, and those 32 are the
        // zero-length buffers, which every decoder is right to accept as an empty stream.
        assert!(tally.err > 10_000, "{tally:?}");
    }

    /// Valid streams with bytes flipped. This reaches deeper into each decoder than random noise
    /// does — the header still parses, so the damage lands in the state machine.
    #[test]
    fn mutated_streams_never_panic() {
        let mut r = Rng::new(0xC0FF_EE01);
        let mut tally = Tally::default();
        for (_, bytes) in corpus() {
            for _ in 0..500 {
                let mut m = bytes.clone();
                for _ in 0..1 + r.below(6) {
                    let at = r.below(m.len() as u32) as usize;
                    m[at] ^= (r.next_u32() & 0xFF) as u8;
                }
                decode_every_way(&m, &mut tally);
            }
        }
        // Both outcomes, in bulk: damaged files that still decode (the mutation landed in a
        // coordinate) and damaged files that are refused (it landed in an opcode or a length).
        // Measured on this seed: 1,021 successes, 23,479 refusals, 122,353 events decoded.
        assert!(tally.ok > 200, "{tally:?}");
        assert!(tally.err > 1_000, "{tally:?}");
        assert!(tally.events > 50_000, "{tally:?}");
    }

    // -----------------------------------------------------------------------------------------
    // (d) Malformed input is reported with its offset.
    // -----------------------------------------------------------------------------------------

    /// `EVT` 2.0 leaves ten of its sixteen opcodes undefined, and each must be refused by offset.
    #[test]
    fn evt2_undefined_opcodes_are_reported_with_their_offset() {
        for code in [0x2u32, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xB, 0xC, 0xD] {
            let mut s = Evt2::encode(&[AerEvent {
                t: 10,
                x: 1,
                y: 2,
                polarity: Polarity::Off,
            }])
            .unwrap();
            let bad = (code << 28).to_le_bytes();
            s.extend_from_slice(&bad);
            let at = s.len() - 4;
            match Evt2::decode(&s) {
                Err(DecodeError::BadOpcode { offset, code: c }) => {
                    assert_eq!(offset, at);
                    assert_eq!(u32::from(c), code);
                }
                other => panic!("opcode 0x{code:X} was not refused: {other:?}"),
            }
        }
    }

    /// `EVT` 3.0 leaves five opcodes undefined. A stream that has drifted out of 16-bit alignment
    /// produces exactly these, so refusing them is what turns a silent desync into an error.
    #[test]
    fn evt3_undefined_opcodes_are_reported_with_their_offset() {
        for code in [0x1u16, 0x9, 0xB, 0xC, 0xD] {
            let mut s = evt3_words(&[0x8000, 0x6000, 0x0001, 0x2001]);
            s.extend_from_slice(&(code << 12).to_le_bytes());
            let at = s.len() - 2;
            match Evt3::decode(&s) {
                Err(DecodeError::BadOpcode { offset, code: c }) => {
                    assert_eq!(offset, at);
                    assert_eq!(u16::from(c), code);
                }
                other => panic!("opcode 0x{code:X} was not refused: {other:?}"),
            }
        }
    }

    /// A stream resumed from an arbitrary offset starts with a column word and no row. That is a
    /// refusal, not an event at row zero.
    #[test]
    fn evt3_refuses_a_column_before_a_row_and_a_mask_before_a_base() {
        let s = evt3_words(&[0x8000, 0x6000, 0x2001]);
        assert!(matches!(Evt3::decode(&s), Err(DecodeError::ColumnBeforeRow { offset: 4 })));
        let s = evt3_words(&[0x8000, 0x6000, 0x0005, 0x4000 | 0xFFF]);
        assert!(matches!(Evt3::decode(&s), Err(DecodeError::VectorBeforeBase { offset: 6 })));
    }

    /// Bit 11 of a row word marks a second camera multiplexed into the stream. Merging the two into
    /// one coordinate space would produce a complete, sorted, wrong event list, so it is refused.
    #[test]
    fn evt3_refuses_a_second_cameras_events_rather_than_merging_them() {
        // ADDR_Y is opcode 0x0, so the word is just the master/slave bit and the row.
        let s = evt3_words(&[0x8000, 0x6000, 0x800 | 5, 0x2001]);
        assert!(matches!(Evt3::decode(&s), Err(DecodeError::UnsupportedSystemType { offset: 4 })));
    }

    /// A trailing odd byte is a truncated word, named by offset.
    #[test]
    fn evt3_reports_a_trailing_half_word() {
        let mut s = evt3_words(&[0x8000, 0x6000, 0x0001, 0x2001]);
        s.push(0x20);
        assert!(matches!(
            Evt3::decode(&s),
            Err(DecodeError::Truncated { offset: 8, need: 2, have: 1 })
        ));
    }

    /// Feeding a file to the wrong decoder must fail on the magic, not produce plausible events.
    #[test]
    fn a_missing_magic_stops_one_format_being_read_as_another() {
        let e = synth(50, 1, 128, 128, 100);
        let evt3 = Evt3::encode(&e, true).unwrap();
        assert!(matches!(
            Aedat2::decode(&evt3, Aedat2Layout::DVS128),
            Err(DecodeError::BadMagic { offset: 0, .. })
        ));
        assert!(matches!(Flat::decode(&evt3), Err(DecodeError::BadMagic { offset: 0, .. })));
        assert!(matches!(Aedat4::decode(&evt3), Err(DecodeError::BadMagic { .. })));
        // A .dat file has no magic at all, only a `%` header that may be absent, so it fails on
        // the record descriptor instead. Stated here because it is a real weakness of that format.
        assert!(matches!(Dat::decode(&evt3), Err(DecodeError::UnsupportedRecordLayout { .. })));
    }

    // -----------------------------------------------------------------------------------------
    // Refusals: the answers that do not exist.
    // -----------------------------------------------------------------------------------------

    fn aedat2_bytes(records: &[(u32, u32)]) -> Vec<u8> {
        let mut v = b"#!AER-DAT2.0\r\n".to_vec();
        for &(a, t) in records {
            v.extend_from_slice(&a.to_be_bytes());
            v.extend_from_slice(&t.to_be_bytes());
        }
        v
    }

    /// The 32-bit microsecond counter wraps after 71.6 minutes, and reading it as an unsigned
    /// pattern with an accumulator recovers the real time. 0xFFFFFFF0 then 0x00000005 is
    /// 4,294,967,280 us then 4,294,967,301 us, not a 71-minute step backwards.
    #[test]
    fn aedat2_unwraps_a_timestamp_across_the_32_bit_wrap() {
        let events = vec![
            AerEvent { t: 0xFFFF_FFF0, x: 1, y: 2, polarity: Polarity::Off },
            AerEvent { t: 0x1_0000_0005, x: 3, y: 4, polarity: Polarity::On },
        ];
        let bytes = Aedat2::encode(&events, Aedat2Layout::DVS128, &[]).unwrap();
        let back = Aedat2::decode(&bytes, Aedat2Layout::DVS128).unwrap();
        assert_eq!(back.events, events);
        assert_eq!(back.events[1].t - back.events[0].t, 21);
    }

    /// A SMALL decrease is not a wrap. Adding 71.6 minutes to the rest of a recording on the
    /// strength of one jittered record is the failure this refusal exists to prevent.
    #[test]
    fn a_small_timestamp_decrease_is_refused_rather_than_treated_as_a_wrap() {
        let bytes = aedat2_bytes(&[(0, 1_000), (0, 900)]);
        match Aedat2::decode(&bytes, Aedat2Layout::DVS128) {
            Err(DecodeError::NonMonotonicTimestamp { offset, previous, found }) => {
                assert_eq!((offset, previous, found), (14 + 8 + 4, 1_000, 900));
            }
            other => panic!("a backwards timestamp was accepted: {other:?}"),
        }
        // The same rule in the .dat decoder.
        let mut d = b"%x\n".to_vec();
        d.extend_from_slice(&[0x00, 0x08]);
        for t in [1_000u32, 900] {
            d.extend_from_slice(&t.to_le_bytes());
            d.extend_from_slice(&0u32.to_le_bytes());
        }
        assert!(matches!(Dat::decode(&d), Err(DecodeError::NonMonotonicTimestamp { .. })));
    }

    /// Encoding refuses what the wire cannot hold, naming the event and the bound.
    #[test]
    fn the_encoders_refuse_rather_than_truncating_a_field() {
        let too_wide = [AerEvent { t: 0, x: 2_048, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Evt2::encode(&too_wide),
            Err(EncodeError::FieldOutOfRange { index: 0, field: "column", value: 2_048, max: 2_047 })
        ));
        assert!(matches!(Evt3::encode(&too_wide, false), Err(EncodeError::FieldOutOfRange { .. })));

        let unsorted = [
            AerEvent { t: 10, x: 0, y: 0, polarity: Polarity::On },
            AerEvent { t: 9, x: 0, y: 0, polarity: Polarity::On },
        ];
        assert!(matches!(
            Flat::encode(&unsorted, 0, 0),
            Err(EncodeError::Unsorted { index: 1, previous: 10, found: 9 })
        ));

        let newline = vec!["a\nb".to_string()];
        assert!(matches!(
            Dat::encode(&[], &newline, 0),
            Err(EncodeError::HeaderLineContainsNewline { index: 0 })
        ));
    }

    /// `EVT` 3.0 carries 24 bits of time. A first timestamp past that, or a gap past that, simply
    /// cannot be written — the wrap count is not on the wire — so the encoder refuses rather than
    /// producing a stream that decodes to a different recording.
    #[test]
    fn evt3_refuses_a_time_the_wire_cannot_express() {
        let late = [AerEvent { t: Evt3::WRAP_US, x: 0, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Evt3::encode(&late, false),
            Err(EncodeError::FieldOutOfRange { field: "first timestamp", .. })
        ));
        let gap = [
            AerEvent { t: 0, x: 0, y: 0, polarity: Polarity::On },
            AerEvent { t: Evt3::WRAP_US, x: 1, y: 0, polarity: Polarity::On },
        ];
        assert!(matches!(Evt3::encode(&gap, false), Err(EncodeError::GapTooLarge { index: 1, .. })));
        // And the guaranteed-safe bound really is safe, at every phase within a window.
        for phase in [0u64, 1, 2_047, 4_095] {
            let ok = [
                AerEvent { t: phase, x: 0, y: 0, polarity: Polarity::On },
                AerEvent { t: phase + Evt3::MAX_GAP_US, x: 1, y: 0, polarity: Polarity::On },
            ];
            let bytes = Evt3::encode(&ok, false).expect("MAX_GAP_US is encodable at any phase");
            assert_eq!(Evt3::decode(&bytes).unwrap().events, ok, "phase {phase}");
        }
    }

    /// `EVT` 2.0's 34 bits of time, at both ends.
    #[test]
    fn evt2_spans_its_whole_34_bit_time_range() {
        let events = vec![
            AerEvent { t: 0, x: 0, y: 0, polarity: Polarity::Off },
            AerEvent { t: 63, x: 2_047, y: 2_047, polarity: Polarity::On },
            AerEvent { t: 64, x: 1, y: 1, polarity: Polarity::Off },
            AerEvent { t: Evt2::MAX_TIME_US, x: 2_047, y: 0, polarity: Polarity::On },
        ];
        let bytes = Evt2::encode(&events).unwrap();
        assert_eq!(Evt2::decode(&bytes).unwrap().events, events);
        let past = [AerEvent { t: Evt2::MAX_TIME_US + 1, x: 0, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Evt2::encode(&past),
            Err(EncodeError::FieldOutOfRange { field: "timestamp", .. })
        ));
    }

    /// A self-contradictory address layout is caught before any byte is read.
    #[test]
    fn an_overlapping_address_layout_is_refused() {
        let bad = Aedat2Layout { p_shift: 2, ..Aedat2Layout::DVS128 };
        assert_ne!(bad.overlap(), 0);
        assert!(matches!(
            Aedat2::decode(&aedat2_bytes(&[]), bad),
            Err(DecodeError::LayoutFieldsOverlap { .. })
        ));
        assert!(matches!(Aedat2::encode(&[], bad, &[]), Err(EncodeError::FieldOutOfRange { .. })));
        assert_eq!(Aedat2Layout::DVS128.overlap(), 0);
        assert_eq!(Aedat2Layout::DAVIS346.overlap(), 0);
    }

    /// A `DAVIS346` special-address word becomes a marker, not a pixel at an invented coordinate.
    #[test]
    fn aedat2_special_words_become_markers_rather_than_pixels() {
        let bytes = aedat2_bytes(&[(1 << 31, 500), (0, 600)]);
        let f = Aedat2::decode(&bytes, Aedat2Layout::DAVIS346).unwrap();
        assert_eq!(f.events.len(), 1);
        assert_eq!(f.markers.len(), 1);
        assert_eq!(f.markers[0].kind, MarkerKind::Other);
        assert_eq!((f.markers[0].offset, f.markers[0].t), (14, 500));
    }

    /// Trigger words survive decoding with their offset and their reconstructed time.
    #[test]
    fn trigger_words_are_reported_not_swallowed() {
        let s = evt3_words(&[0x8001, 0x6002, 0xA000 | 1, 0x0003, 0x2004]);
        let f = Evt3::decode(&s).unwrap();
        assert_eq!(f.events.len(), 1);
        assert_eq!(f.markers.len(), 1);
        assert_eq!(f.markers[0].kind, MarkerKind::ExternalTrigger);
        assert_eq!((f.markers[0].offset, f.markers[0].t), (4, 4_098));
        let mut e2 = Evt2::encode(&[AerEvent { t: 100, x: 1, y: 1, polarity: Polarity::On }])
            .unwrap();
        e2.extend_from_slice(&(0xAu32 << 28).to_le_bytes());
        assert_eq!(Evt2::decode(&e2).unwrap().markers[0].kind, MarkerKind::ExternalTrigger);
    }

    /// The flat format's three checks: magic, declared count, and the reserved bytes.
    #[test]
    fn flat_checks_its_own_header_and_reserved_bytes() {
        let e = synth(4, 3, 64, 64, 10);
        let good = Flat::encode(&e, 64, 64).unwrap();

        let mut short = good.clone();
        short.truncate(good.len() - 4);
        assert!(matches!(Flat::decode(&short), Err(DecodeError::CountMismatch { offset: 16, .. })));

        let mut extra = good.clone();
        extra.push(0);
        assert!(matches!(Flat::decode(&extra), Err(DecodeError::CountMismatch { .. })));

        let mut dirty = good.clone();
        dirty[Flat::HEADER_SIZE + 14] = 1;
        assert!(matches!(
            Flat::decode(&dirty),
            Err(DecodeError::ReservedNotZero { offset: 38 })
        ));

        let mut pol = good.clone();
        pol[Flat::HEADER_SIZE + 12] = 2;
        assert!(matches!(
            Flat::decode(&pol),
            Err(DecodeError::FieldOutOfRange { field: "polarity", value: 2, max: 1, .. })
        ));
    }

    /// A `.dat` descriptor this decoder does not handle is named, with its two bytes.
    #[test]
    fn dat_reports_a_record_layout_it_cannot_read() {
        let mut d = b"% Width 64\n".to_vec();
        d.extend_from_slice(&[0x07, 0x10]);
        match Dat::decode(&d) {
            Err(DecodeError::UnsupportedRecordLayout { offset, record_type, record_size }) => {
                assert_eq!((offset, record_type, record_size), (11, 0x07, 0x10));
            }
            other => panic!("an unknown record layout was accepted: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------------------------
    // AEDAT 4.0: the framing works, the compressed payload refuses, and says which.
    // -----------------------------------------------------------------------------------------

    /// The capability boundary of a zero-dependency crate, asserted rather than described.
    ///
    /// A compressed file still yields its packet framing — stream ids, sizes, offsets — and only
    /// [`Aedat4::events`] refuses, naming the compression and the packet's offset. And the file
    /// survives a decode-encode cycle byte for byte even though nothing read inside it.
    #[test]
    fn aedat4_decodes_the_framing_of_a_compressed_file_and_refuses_its_events() {
        let mut f = b"#!AER-DAT4.0\r\n#Format: COMPRESSED_LZ4\r\n#!END-HEADER\r\n".to_vec();
        f.extend_from_slice(&3i32.to_le_bytes());
        f.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        for (id, payload) in [(0i32, vec![1u8, 2, 3, 4]), (1, vec![9u8; 7])] {
            f.extend_from_slice(&id.to_le_bytes());
            f.extend_from_slice(&i32::try_from(payload.len()).unwrap().to_le_bytes());
            f.extend_from_slice(&payload);
        }
        let d = Aedat4::decode(&f).expect("the framing decodes whatever the compression");
        assert_eq!(d.compression, Aedat4Compression::Lz4);
        assert_eq!(d.io_header, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(d.packets.len(), 2);
        assert_eq!(d.packets[1].stream_id, 1);
        assert_eq!(d.packets[1].payload, vec![9u8; 7]);
        assert!(d.packets.iter().all(|p| p.events.is_none()));
        match d.events() {
            Err(DecodeError::UnsupportedCompression { offset, name }) => {
                assert_eq!(name, "COMPRESSED_LZ4");
                assert_eq!(offset, d.packets[0].offset);
            }
            other => panic!("a compressed payload was not refused: {other:?}"),
        }
        assert_eq!(d.encode().unwrap(), f, "a file we cannot read inside still re-saves exactly");
    }

    /// A payload whose `FlatBuffers` offsets point outside it is refused, not followed.
    #[test]
    fn aedat4_refuses_a_flatbuffer_that_points_outside_itself() {
        let e = synth(8, 11, 64, 64, 5);
        let file = Aedat4::raw_from_events(&e, 8, 0).unwrap();
        let good = file.encode().unwrap();
        let decoded = Aedat4::decode(&good).expect("valid to start with");
        // The payload begins 8 bytes past the packet header, and its first four bytes are the
        // FlatBuffers root offset. Point it past the end of the payload.
        let root_at = decoded.packets[0].offset + 8;
        let mut bad = good.clone();
        bad[root_at] = 0xFF;
        bad[root_at + 1] = 0xFF;
        match Aedat4::decode(&bad) {
            Err(DecodeError::MalformedFlatBuffer { .. } | DecodeError::CountMismatch { .. }) => {}
            Err(other) => panic!("wrong refusal: {other:?}"),
            Ok(_) => panic!("a FlatBuffer pointing outside itself was followed"),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Into the rest of the crate.
    // -----------------------------------------------------------------------------------------

    /// Addresses, polarity planes and the tick division, each against hand arithmetic.
    #[test]
    fn the_train_map_addresses_pixels_the_way_it_says_it_does() {
        let m = TrainMap { width: 640, height: 480, tick_us: 1_000, split_polarity: true };
        assert_eq!(m.address_count(), Some(640 * 480 * 2));
        let events = vec![
            AerEvent { t: 0, x: 3, y: 2, polarity: Polarity::Off },
            AerEvent { t: 999, x: 3, y: 2, polarity: Polarity::On },
            AerEvent { t: 1_000, x: 0, y: 0, polarity: Polarity::Off },
        ];
        let train = m.map(&events).expect("in range");
        let s = train.spikes();
        assert_eq!(s.len(), 3);
        // y * width + x = 2 * 640 + 3 = 1283; the On plane is offset by 640 * 480 = 307,200.
        assert_eq!((s[0].t, s[0].source), (0, 1_283));
        assert_eq!((s[1].t, s[1].source), (0, 1_283 + 307_200), "999 us floors into tick 0");
        assert_eq!((s[2].t, s[2].source), (1, 0));

        // Without the split, the two planes collapse and the contrast direction is gone.
        let merged = TrainMap { split_polarity: false, ..m };
        assert_eq!(merged.address_count(), Some(640 * 480));
        let mt = merged.map(&events).unwrap();
        assert_eq!(mt.spikes()[0].source, mt.spikes()[1].source);

        // Refusals rather than clamps.
        let out = [AerEvent { t: 0, x: 640, y: 0, polarity: Polarity::On }];
        assert!(m.map(&out).is_none());
        assert!(TrainMap { tick_us: 0, ..m }.map(&events).is_none());
    }

    /// The flattening and unflattening are inverses, and both refuse outside the geometry.
    #[test]
    fn flattening_to_a_crate_event_is_invertible() {
        for (x, y) in [(0u16, 0u16), (639, 479), (17, 23)] {
            let e = AerEvent { t: 42, x, y, polarity: Polarity::On };
            let flat = e.to_event(640).expect("in range");
            assert_eq!(flat.address, u32::from(y) * 640 + u32::from(x));
            assert_eq!(AerEvent::from_event(flat, 640), Some(e));
        }
        assert!(AerEvent { t: 0, x: 640, y: 0, polarity: Polarity::On }.to_event(640).is_none());
        assert!(AerEvent { t: 0, x: 0, y: 0, polarity: Polarity::On }.to_event(0).is_none());
    }

    /// The errors are printable, carry their offsets in the text, and cross a `Box<dyn Error>`
    /// boundary — which is what stops a caller reaching for `.unwrap()`.
    #[test]
    fn errors_name_the_offset_in_their_text_and_box_cleanly() {
        let e = DecodeError::BadOpcode { offset: 1_234, code: 0xD };
        let text = e.to_string();
        assert!(text.contains("1234"), "{text}");
        assert!(text.contains("0xD"), "{text}");
        let _boxed: Box<dyn std::error::Error> = Box::new(e);
        let f = EncodeError::GapTooLarge { index: 7, gap: 99, max: 10 };
        assert!(f.to_string().contains("event 7"), "{}", f.to_string());
        let _boxed2: Box<dyn std::error::Error> = Box::new(f);
    }

    /// A real `EVT` 2.0 sensor emits a `TIME_HIGH` every 64 microseconds whether or not the value
    /// changed. A repeat must not be mistaken for time running backwards.
    ///
    /// Constructed by hand because this crate's encoder emits the minimum number of time words and
    /// so never produces the case it has to survive.
    #[test]
    fn evt2_accepts_a_repeated_time_high_as_real_sensors_emit_it() {
        let mut s = Vec::new();
        let mut push = |w: u32| s.extend_from_slice(&w.to_le_bytes());
        push(0x8000_000A); // TIME_HIGH 10 -> base 640 us
        push((1 << 28) | (63 << 22) | (5 << 11) | 6); // t = 703
        push(0x8000_000A); // the SAME TIME_HIGH again; base is 640, 63 us behind the last event
        push((1 << 28) | (63 << 22) | (7 << 11) | 8); // t = 703 again
        push(0x8000_000B); // TIME_HIGH 11 -> base 704
        push((1 << 22) | (9 << 11) | 10); // CD_OFF is opcode 0x0, so: t = 705, falling
        let got = Evt2::decode(&s).expect("a repeated TIME_HIGH is not an error");
        assert_eq!(got.events.iter().map(|e| e.t).collect::<Vec<_>>(), vec![703, 703, 705]);
        assert_eq!(got.events[2].polarity, Polarity::Off);
    }

    /// A run with holes in it wider than one mask: the encoder must plant a new `VECT_BASE_X`
    /// rather than emit empty masks across the gap, and the decoder must follow it.
    ///
    /// Columns 100, 101, then 130, 131, 132, then 200 in one microsecond on one row. The gap from
    /// 101 to 130 is 29 columns, wider than a 12-bit mask, so a re-base is forced twice.
    #[test]
    fn evt3_rebases_a_vector_run_across_a_gap_wider_than_a_mask() {
        let events: Vec<AerEvent> = [100u16, 101, 130, 131, 132, 200]
            .iter()
            .map(|&x| AerEvent { t: 7_000, x, y: 42, polarity: Polarity::On })
            .collect();
        let bytes = Evt3::encode(&events, true).expect("encodable");
        let back = Evt3::decode(&bytes).expect("decodable");
        assert_eq!(back.events, events);
        // Three time/row words, three VECT_BASE_X and three masks: 9 words. An encoder that padded
        // across the gap with empty masks would need 12, and one that gave up on the vector path
        // would emit 6 column words. Asserted so that a regression in either direction shows.
        assert_eq!(bytes.len(), 9 * 2, "{} bytes", bytes.len());
    }

    /// The round trip on a stream dense enough that the vector path actually runs: 200 bursts of
    /// up to 40 same-microsecond events on one row, which is what an edge crossing the array looks
    /// like and what the random stream above almost never produces.
    #[test]
    fn evt3_round_trips_a_dense_vectorisable_stream() {
        let mut r = Rng::new(0x5151);
        let mut events = Vec::new();
        let mut t = 0u64;
        for _ in 0..200 {
            t += u64::from(r.below(900));
            let y = r.below(720) as u16;
            let polarity = if r.next_u32() & 1 == 0 { Polarity::Off } else { Polarity::On };
            let mut x = r.below(600) as u16;
            for _ in 0..1 + r.below(40) {
                events.push(AerEvent { t, x, y, polarity });
                x += 1 + r.below(3) as u16;
            }
        }
        let plain = Evt3::encode(&events, false).expect("encodable");
        let vect = Evt3::encode(&events, true).expect("encodable");
        assert_eq!(Evt3::decode(&plain).unwrap().events, events);
        assert_eq!(Evt3::decode(&vect).unwrap().events, events);
        assert!(vect.len() < plain.len(), "vect {} vs plain {}", vect.len(), plain.len());
    }
}
