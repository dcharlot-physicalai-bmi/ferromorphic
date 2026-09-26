//! Address-event representation: the wire formats event sensors actually speak.
//!
//! # What `AER` is, and what it buys
//!
//! A conventional image sensor is read out on a clock: every pixel reports every frame, whether or
//! not anything happened to it. Misha Mahowald's address-event representation (M. Mahowald, *VLSI
//! analogs of neuronal visual processing: a synthesis of form and function*, doctoral dissertation,
//! California Institute of Technology (1992), doi:10.7907/4bdw-fg34; and, at the same lab, M. A.
//! Sivilotti, *Wiring considerations in analog VLSI systems, with application to
//! field-programmable networks*, doctoral dissertation, California Institute of Technology (1991),
//! doi:10.7907/stj4-kh72) replaces that with a **shared digital bus that carries the identity of
//! whichever element just fired**. A 128x128 array does not need 16,384 wires; it needs 14 address
//! lines and an arbiter. The spike itself is not transmitted — its *address* is, and its *time* is
//! the moment the transmission happens.
//!
//! This paragraph used to give Mahowald's thesis as *VLSI Phototransduction and Stereopsis* and
//! Sivilotti's only as "contemporaneous work at the same lab". `DataCite` records the 1992
//! dissertation under the title above, and this review did not locate any work by the old title: a
//! `DataCite` title search for "phototransduction" and "stereopsis" together returns nothing, and
//! the nearest Mahowald work `Crossref` holds is the 1994 book *An Analog VLSI System for
//! Stereoscopic Vision*, doi:10.1007/978-1-4615-2724-4.
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
//! *Rollover.* The low bits wrap. `EVT` 3.0 puts 12 bits of time in an `EVT_TIME_LOW` word and 12
//! more in an `EVT_TIME_HIGH` word — no event word carries time at all; an event takes whatever
//! base the two time words last set, which is the whole point of the encoding — so the wire carries
//! 24 bits — 16.777216 seconds — and a recording longer than
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
//!
//!    **On 32 bits too, which is where the claim was thin.** A `FlatBuffers` payload's offsets are
//!    `u32`s the file chose, and `usize` on `wasm32` is 32 bits, so adding them wraps there and
//!    the wrapped value passes the bounds check that follows. [`Aedat4`] therefore does that
//!    arithmetic in `u64`, which makes the 32-bit and the 64-bit path the same path. The whole
//!    battery, fuzz tests included, was run under `wasm32-wasip1` with overflow checks on: 60
//!    tests, all green, and `aedat4_refuses_offsets_that_would_wrap_a_32_bit_usize` fails there if
//!    the arithmetic is narrowed back to `usize`.
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
//!
//! Coordinates are pixels: `x` to the right, `y` down, origin top-left, in every event a decoder
//! here **returns**. That is not the same as on the wire, and this paragraph used to say it was —
//! that all four formats put the origin top-left on the wire. They do not. iniVation's `AEDAT` 1.0
//! and 2.0 pages both say of the row field "(0, 0) in the lower left corner of the screen";
//! `jAER`'s extractors take that field as it stands (`e.y = (short) ((addr & YMASK) >>> YSHIFT)`)
//! and so leave row 0 where the pages put it; and iniVation's own `dv-processing` parser for the
//! `DVS128` turns it round by default, "Invert Y values (flip along Y axis). To convert to CG
//! format." The `AEDAT` 3.1 page, by contrast, says "(0, 0) in upper left corner of screen". The
//! `AEDAT` 2.0 presets therefore set [`Aedat2Layout::y_invert`], so that a row from a `jAER`
//! recording counts from the top like every other decoder's; through 0.22.0 they did not, and such
//! a recording came out upside down against the other formats. The same two pages say the same
//! words of the column field, but `jAER`'s extractors mirror that field (`sxm - ...` for the
//! `DVS128`, `sx1 - ...` for a `DAVIS`) and `dv-processing` mirrors it for the `DVS128` "To correct
//! for flipped camera"; the presets follow the code, and each says so.
//! Whether a given sensor's *optics* invert the image is a property of the camera, not of the
//! file, and this module does not attempt to correct it.
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
//! the doc of the item concerned** rather than presenting a guess confidently. The two known soft
//! spots are the polarity bit's sense in `AEDAT` 2.0 ([`Aedat2Layout::p_on_is_one`]), where for the
//! `DVS128` iniVation's documentation and iniVation's code disagree, and the `.dat` record-type byte
//! ([`Dat::CD_TYPE_CODES`]). There used to be a third, the `FlatBuffers` table inside an `AEDAT` 4.0
//! packet, which this implementation had not checked against a file written by `DV` software. When
//! it was, the container around the table turned out to be wrong as well, and no file `DV` writes
//! could be opened: see [`Aedat4`].

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
    /// `TIME_HIGH` word reads zero. That is the honest answer: the stream did not say. The same is
    /// true of an [`AerEvent`]: an `EVT` 3.0 event word that arrives before any time word decodes
    /// at `t = 0`, because an event word carries no time of its own and the base has not been set.
    /// A stream resumed from an arbitrary offset is the usual way to see this, and the first row
    /// or column word of one is more often [`DecodeError::ColumnBeforeRow`] — the time is the part
    /// the format cannot detect the absence of. `evt3_dates_an_event_before_any_time_word_at_zero`
    /// pins the behaviour so it cannot change silently.
    pub t: u64,
}

/// Which family of non-pixel word a [`Marker`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MarkerKind {
    /// An external trigger pulse — `EXT_TRIGGER` in `Prophesee`'s formats, and the external-input
    /// word of a `DAVIS` in `AEDAT` 2.0. This is the wire's synchronisation channel.
    ///
    /// [`Aedat2`] produces this variant only through [`Aedat2Layout::trigger_mask`], which only the
    /// [`Aedat2Layout::DAVIS346`] preset sets, on the authority that preset names: `jAER` and
    /// iniVation's `AEDAT` 2.0 page agree on which `DAVIS` word is the external input. A special
    /// word under a layout with no trigger mask is [`MarkerKind::Other`], which claims only that
    /// the word is not a pixel.
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
    /// The file did not begin with the magic this decoder requires, or a `FlatBuffers` buffer inside
    /// an `AEDAT` 4.0 file did not carry the file identifier expected of it (`IOHE`, `EVTS`).
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
        /// The compression the file's `IOHeader` declares, by the name its `CompressionType` enum
        /// gives it (`LZ4`, `ZSTD_HIGH`), or by its number when the schema names no such value.
        /// This used to be a `Format:` header line's text, which is not part of `AEDAT` 4.0.
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

/// Why a stream could not be encoded. Every variant about an event names the **index of the
/// event** at fault.
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
    /// An `AEDAT` 4.0 `IOHeader` puts the file data table somewhere other than where it would be
    /// written.
    ///
    /// The table follows the last packet, and it indexes the packets by byte offset, so it is
    /// stale the moment a packet changes size. [`Aedat4::encode`] refuses rather than write an
    /// `IOHeader` that points into the middle of a packet, or a table that no `IOHeader` points
    /// to; [`Aedat4::drop_data_table`] is the way out.
    DataTableMisplaced {
        /// The `IOHeader`'s `dataTablePosition`, bytes from the start of the file, or `None` when
        /// it is -1, "no table present".
        declared: Option<u64>,
        /// Where the packets being written end, which is the only place the table can begin.
        packets_end: u64,
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
            Self::DataTableMisplaced { declared: Some(at), packets_end } => write!(
                f,
                "the IOHeader places the file data table at byte {at}, but the packets end at byte {packets_end}; drop the table or restore the packets"
            ),
            Self::DataTableMisplaced { declared: None, packets_end } => write!(
                f,
                "the IOHeader declares no file data table, but one would be written at byte {packets_end}"
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

/// A header line's bytes as text, or `None` when they are not text and so are not a header line.
///
/// Text means valid `UTF-8` with no control character other than a horizontal tab. That admits
/// every real `jAER` or `Prophesee` comment line, including a non-`ASCII` one, and excludes the
/// binary records that would otherwise be eaten as comments — see [`read_header_lines`].
fn header_text(b: &[u8]) -> Option<String> {
    let s = core::str::from_utf8(b).ok()?;
    if s.chars().any(|c| c.is_control() && c != '\t') {
        return None;
    }
    Some(s.to_string())
}

/// Read `#`- or `%`-prefixed text header lines from the front of a buffer.
///
/// Returns the lines with the prefix and the line terminator removed, and the offset of the first
/// byte after the header.
///
/// # Where the header stops, and why that needs deciding at all
///
/// Neither `AEDAT` 2.0 nor `.dat` terminates its header: the binary record array begins at the
/// first byte that is not a header line, so the decoder has to tell the two apart from the bytes.
/// Taking "begins with the prefix byte" as sufficient **loses records**, because a record can begin
/// with that byte: a `DAVIS346` row field of 140 to 143 puts `0x23`, the character `#`, in the
/// big-endian address MSB, and the record is then swallowed as a comment line, taking every byte
/// up to the next `0x0A` with it. Four of that sensor's 260 rows do this — decoded rows 116 to 119,
/// since the field counts from the bottom edge and [`Aedat2Layout::DAVIS346`] turns it round.
///
/// So a candidate line is accepted as a header line only if it is **text** by [`header_text`]. A
/// binary record essentially always carries a control byte — the zero high byte of a coordinate or
/// of a young timestamp — before the next newline, and a candidate that is not text ends the
/// header and becomes the first record instead. A *text* line with no newline at all is
/// [`DecodeError::Truncated`] rather than a silently accepted final line, because the missing
/// newline is exactly what a cut-short file looks like.
///
/// This is a heuristic over formats that offer the reader nothing better, and it is stated as one.
/// What is proved rather than hoped: `aedat2_round_trips_every_first_event_row_of_both_presets`
/// sweeps every row of both shipped presets and shows that no address either preset can write is
/// mistaken for a comment.
fn read_header_lines(b: &[u8], prefix: u8) -> Result<(Vec<String>, usize), DecodeError> {
    let mut lines = Vec::new();
    let mut at = 0usize;
    while b.get(at) == Some(&prefix) {
        let start = at + 1;
        let nl = b[start..].iter().position(|&c| c == b'\n');
        let text_end = match nl {
            Some(n) => {
                let e = start + n;
                if e > start && b[e - 1] == b'\r' { e - 1 } else { e }
            }
            None => b.len(),
        };
        let Some(text) = header_text(&b[start..text_end]) else { break };
        let Some(n) = nl else {
            return Err(DecodeError::Truncated {
                offset: at,
                need: b.len() - at + 1,
                have: b.len() - at,
            });
        };
        lines.push(text);
        at = start + n + 1;
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

/// Refuse a stream whose timestamps a 32-bit wrapping clock could not be unwrapped back to.
///
/// Shared by [`Aedat2`] and [`Dat`], which write the same truncated `uint32` microsecond counter
/// and recover it with the same "a backwards jump of more than half the range is a wrap" rule.
/// Two things defeat that rule and both are refused here rather than written:
///
/// * a first timestamp at or past `max_time`, because the decoder's accumulator starts at zero and
///   the dropped high bits are nowhere on the wire;
/// * a gap past `max_gap` — half the counter's range — because a backwards jump of exactly half is
///   ambiguous, and a jump of less than half reads as a file whose time runs backwards. A gap of a
///   whole counter period reads as no gap at all: measured, two events 2^32 us apart came back
///   0 us apart, in order, with no error.
fn check_wrapping_clock(
    events: &[AerEvent],
    max_time: u64,
    max_gap: u64,
) -> Result<(), EncodeError> {
    if let Some(first) = events.first() {
        fit(0, "first timestamp", first.t, max_time)?;
    }
    for i in 1..events.len() {
        // Saturating rather than plain: `check_sorted` runs first at both call sites, and a
        // subtraction that depends on a caller elsewhere having done its job is a panic waiting
        // for the edit that reorders them.
        let gap = events[i].t.saturating_sub(events[i - 1].t);
        if gap > max_gap {
            return Err(EncodeError::GapTooLarge { index: i, gap, max: max_gap });
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
    /// Whether the column counts from the right edge: the stored value is `width - 1 - x`, the
    /// mirror `jAER`'s extractors apply, or `2^x_bits - 1 - x` for a layout that states no width.
    ///
    /// Several `AER` chips wire the column address backwards relative to the optical image. The
    /// transform is its own inverse, so an encode-decode round trip cannot detect a wrong setting;
    /// only a picture of a known scene can.
    ///
    /// Mirrored about the last column of the stated sensor, not about the top of the field: the two
    /// agree only when the sensor fills the field, as 128 columns fill seven bits. Through 0.22.0 the stored value was always
    /// `2^x_bits - 1 - x`, which is right for the 128-column `DVS128` and cannot express the
    /// `DAVIS346`'s mirror at all: about 1023, its 346 columns land on 678 to 1023 and every one
    /// fails the width check. A layout that mirrors a field must state a width or height the field
    /// can hold, or it is refused before a byte is read.
    pub x_invert: bool,
    /// Bit position of the least significant row bit.
    pub y_shift: u32,
    /// Number of row bits.
    pub y_bits: u32,
    /// Whether the row counts from the bottom edge: the stored value is `height - 1 - y`, or
    /// `2^y_bits - 1 - y` for a layout that states no height. Same caveats as
    /// [`Aedat2Layout::x_invert`].
    ///
    /// Both presets set it, because `AEDAT` 2.0 counts rows from the bottom and every decoder in
    /// this module returns them counted from the top; see the module doc.
    pub y_invert: bool,
    /// Bit position of the single polarity bit.
    pub p_shift: u32,
    /// Whether a set polarity bit means [`Polarity::On`].
    ///
    /// **This is the flag this implementation is least sure of.** Readers of the same chips
    /// disagree about the sense of this bit — for the `DVS128`, iniVation's own file-format page
    /// disagrees with iniVation's own parser — and both choices produce an event stream that looks
    /// entirely normal: the scene simply has its contrast inverted, which no event count, rate plot
    /// or timestamp check can detect. It is exposed as a field, and each preset states what this
    /// implementation chose and on whose authority, so that a caller who has a recording of a known
    /// stimulus can settle it for their own data rather than inheriting a guess.
    pub p_on_is_one: bool,
    /// Bits that mark a non-pixel word. A record whose address has any of these bits set becomes a
    /// [`Marker`] instead of an event. Zero disables the check.
    pub special_mask: u32,
    /// The bits of [`Aedat2Layout::special_mask`] that mark an external trigger.
    ///
    /// A marker whose address sets only these of the special bits is a
    /// [`MarkerKind::ExternalTrigger`]; one that sets any other special bit is a
    /// [`MarkerKind::Other`], because on a `DAVIS` the other bit changes what the rest of the word
    /// means. Bits outside `special_mask` have no effect, and zero makes every marker
    /// [`MarkerKind::Other`].
    pub trigger_mask: u32,
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
    /// Row in bits 8-14, column in bits 1-7, polarity in bit 0, both coordinates counted from the
    /// far edge. Transcribed from the `jAER` `Tmpdiff128` extractor as this implementation reads it.
    /// iniVation's `dv-processing` parser for the same chip reads the same address the same way by
    /// default: `x = (WIDTH - 1) - x`, "To correct for flipped camera", and `y = (HEIGHT - 1) - y`,
    /// "To convert to CG format". `jAER` mirrors the column and leaves the row counting from the
    /// bottom; this preset turns the row round as well, which is what makes its output top-left like
    /// every other decoder's. Through 0.22.0 it did not (see the module doc).
    ///
    /// **The polarity sense is "bit 0 clear means `On`", and it is unverified against a physical
    /// device.** iniVation's `AEDAT` 1.0 page says the opposite: "0 | Polarity | Polarity
    /// (luminosity change): '1' means increase (ON), '0' means decrease (OFF)." Two pieces of code
    /// agree with this preset against it: `jAER`'s `DVS128.java` ("The ON events have raw polarity
    /// 0", `e.type = (byte) ((1 - addr) & 1)`) and iniVation's own `dv-processing` `DVS128` parser,
    /// which reads a clear bit as `On` with the comment "Invert polarity bit. Hardware is like this."
    /// This doc used to attribute the opposite reading only to "circulating `Python` readers"; the
    /// vendor's own documentation is the source that disagrees. See
    /// [`Aedat2Layout::p_on_is_one`].
    ///
    /// # Everything above bit 14 is a marker, not a pixel
    ///
    /// The address is 15 bits wide, so bits 15-31 are not part of it. This preset therefore sets
    /// [`Aedat2Layout::special_mask`] to all of them: a record with any bit set above bit 14 is
    /// reported as a [`Marker`] rather than decoded. A `special_mask` of zero — which this preset
    /// carried until 0.5.0 — decodes such a record as an ordinary pixel, so an external-trigger
    /// word became a fabricated event at column 127 and the synchronisation evidence the [`Marker`]
    /// type exists to preserve was destroyed while the event count stayed plausible.
    ///
    /// What is claimed here is only what the 15-bit address width supports: those bits are **not a
    /// pixel**. Which of them `jAER` uses for its sync word this implementation did not confirm
    /// against a device, so the marker's kind is [`MarkerKind::Other`] rather than a guess at
    /// [`MarkerKind::ExternalTrigger`].
    pub const DVS128: Self = Self {
        x_shift: 1,
        x_bits: 7,
        x_invert: true,
        y_shift: 8,
        y_bits: 7,
        y_invert: true,
        p_shift: 0,
        p_on_is_one: false,
        special_mask: !0x7FFF,
        trigger_mask: 0,
        width: 128,
        height: 128,
        source: "DVS128 (Lichtsteiner et al. 2008), jAER Tmpdiff128 extractor",
    };

    /// The `DAVIS346`: 346x260 (Taverni, Moeys, Li, Cavaco, Motsnyi, San Segundo Bello and
    /// Delbruck, *Front and Back Illuminated Dynamic and Active Pixel Vision Sensors Comparison*,
    /// IEEE Trans. Circuits Syst. II 65(5):677-681 (2018), doi:10.1109/TCSII.2018.2824899).
    ///
    /// Row in bits 22-30, column in bits 12-21, polarity in bit 11, external input in bit 10, bit 31
    /// marking an active-pixel-sensor or inertial word. The constants are `jAER`'s `DavisChip.java`
    /// — this doc used to say `DavisBaseCamera`, which holds the extractor that uses them — and
    /// iniVation's `AEDAT` 2.0 page lays out the same bits: "31 | Type | ... '0' means DVS, '1'
    /// means APS or IMU", then for a `DVS` word "11-10 | sub-Type | 00 -> DVS Polarity OFF 01 ->
    /// External Event (same as 11) 10 -> DVS Polarity ON 11 -> External Event (Same as 01)". That
    /// table is also what this preset's polarity sense, bit 11 set means `On`, rests on. **This
    /// implementation did not verify any of it against a `DAVIS` recording**, and the interleaved
    /// active-pixel-sensor samples that share this address space are reported as
    /// [`MarkerKind::Other`] rather than decoded.
    ///
    /// # The column is mirrored, about column 345
    ///
    /// `jAER`'s `DavisEventExtractor` sets `sx1 = getChip().getSizeX() - 1` and decodes a `DVS`
    /// word as `e.x = (short) (sx1 - ((data & DavisChip.XMASK) >>> DavisChip.XSHIFT))`; the
    /// `DAVIS346` has no extractor of its own and a size of 346. Through 0.22.0 this preset set
    /// `x_invert: false`, so its every column was the mirror image of `jAER`'s — and setting it to
    /// `true` would not have fixed that, because inversion then meant `1023 - x` (see
    /// [`Aedat2Layout::x_invert`]). The row is turned round as well, so that it counts from the top.
    ///
    /// # Bit 10 is a trigger, not a pixel
    ///
    /// A `DVS` word with bit 10 set is an external-input event. `jAER`'s `DavisChip.java` names it
    /// `EXTERNAL_INPUT_EVENT_ADDR = 1 << EVENT_TYPE_SHIFT`, "This special address is is for external
    /// pin input events", with bits 0-2 then carrying falling, rising or pulse as 2, 3 or 4, and
    /// its extractor marks such a word special ("if special bit for DVS address (bit 10) is set,
    /// then mark this as spscial event"). Since June 2023 `jAER` still fills in a coordinate and a
    /// polarity for it, for `v2e` noise labelling, but it does not treat it as a pixel. This preset
    /// therefore puts bit 10 in [`Aedat2Layout::special_mask`] and in
    /// [`Aedat2Layout::trigger_mask`], and such a word becomes a [`MarkerKind::ExternalTrigger`]
    /// whose [`Marker::raw`] carries the edge in its low three bits. A word that also sets bit 31
    /// is an active-pixel-sensor or inertial word whose bits 11-10 mean something else, and stays
    /// [`MarkerKind::Other`]. A pixel word that `v2e` labelled by setting bit 10 arrives here as a
    /// trigger marker too, its coordinate fields intact in [`Marker::raw`]: `jAER` marks both uses
    /// special without telling them apart, and this preset does not tell them apart either.
    ///
    /// This doc used to say that bits 0-10 belong to no field, that this implementation had not
    /// located a statement of what a `DAVIS` puts there, and that a mask over them would be a guess;
    /// every trigger was decoded as a pixel. Both sources above are that statement. Bits 0-9 of a
    /// `DVS` word are "10-bit ADC sample ... Only for Type=APS, else zero", and this preset ignores
    /// them, as `jAER` does.
    pub const DAVIS346: Self = Self {
        x_shift: 12,
        x_bits: 10,
        x_invert: true,
        y_shift: 22,
        y_bits: 9,
        y_invert: true,
        p_shift: 11,
        p_on_is_one: true,
        special_mask: (1 << 31) | (1 << 10),
        trigger_mask: 1 << 10,
        width: 346,
        height: 260,
        source: "DAVIS346 (Taverni et al. 2018), jAER DavisChip constants and DavisEventExtractor",
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

    /// The widest value the column and row fields can hold, as they are actually placed.
    ///
    /// `field_mask` clips a field at bit 31, so a 17-bit field at shift 20 really holds 12 bits;
    /// this reports what the layout does, not what it declares. A coordinate is a `u16`, so a field
    /// that can hold more than 65535 has no representable answer and is refused rather than
    /// saturated.
    fn coord_span(&self) -> (u32, u32) {
        (
            Self::field_mask(self.x_shift, self.x_bits) >> self.x_shift,
            Self::field_mask(self.y_shift, self.y_bits) >> self.y_shift,
        )
    }

    fn check(&self) -> Result<(), DecodeError> {
        let mask = self.overlap();
        if mask != 0 {
            return Err(DecodeError::LayoutFieldsOverlap { mask });
        }
        let (x_span, y_span) = self.coord_span();
        for (span, field) in [(x_span, "x_bits"), (y_span, "y_bits")] {
            if u64::from(span) > u64::from(u16::MAX) {
                return Err(DecodeError::FieldOutOfRange {
                    offset: 0,
                    field,
                    value: u64::from(span),
                    max: u64::from(u16::MAX),
                });
            }
        }
        if let Some((field, value, max)) = self.mirror_fault() {
            return Err(DecodeError::FieldOutOfRange { offset: 0, field, value, max });
        }
        Ok(())
    }

    /// A stated width or height that an inverted field cannot be mirrored about, as
    /// `(field, value, max)`.
    ///
    /// An inverted field stores `dimension - 1 - coordinate`, so the last column or row of the
    /// sensor must be a value the field can hold. A width of 2000 over a 10-bit column would
    /// otherwise write column 0 as 1999, which the field truncates to 975: a file that decodes, to
    /// the wrong picture. An uninverted field has no such constraint — the columns past its reach
    /// are merely unreachable — so it is not refused.
    fn mirror_fault(&self) -> Option<(&'static str, u64, u64)> {
        let (x_span, y_span) = self.coord_span();
        [(self.x_invert, self.width, x_span, "width"), (self.y_invert, self.height, y_span, "height")]
            .into_iter()
            .find(|&(invert, dim, span, _)| invert && u64::from(dim) > u64::from(span) + 1)
            .map(|(_, dim, span, field)| (field, u64::from(dim), u64::from(span) + 1))
    }

    /// What an inverted field is mirrored about: the last column or row of the stated sensor, or
    /// the top of the field for a layout that states no geometry.
    fn pivot(m: u32, dim: u16) -> u32 {
        if dim == 0 { m } else { u32::from(dim) - 1 }
    }

    /// The coordinate a field decodes to, or `Err((raw, last))` — the field as it sits on the wire
    /// and the last value the stated sensor has — when it names a column or row past the sensor.
    ///
    /// The range check is made on the raw field, before the mirror, because a mirrored field past
    /// the sensor has no coordinate to report: `jAER` would decode column 346 of a `DAVIS346` as
    /// -1.
    fn get(&self, addr: u32, shift: u32, bits: u32, invert: bool, dim: u16) -> Result<u16, (u32, u32)> {
        let m = Self::field_mask(shift, bits) >> shift;
        let raw = (addr >> shift) & m;
        if dim != 0 && raw >= u32::from(dim) {
            return Err((raw, u32::from(dim) - 1));
        }
        let v = if invert { Self::pivot(m, dim) - raw } else { raw };
        // Unreachable by construction: `Aedat2Layout::check` refuses, before a byte is read, any
        // layout whose column or row field can hold more than `u16::MAX`, and `v` is at most the
        // larger of `m` and `dim - 1`. The saturation is what a `u16` conversion must do with no
        // `Result` to return, and
        // `a_coordinate_field_wider_than_a_u16_is_refused_before_any_byte_is_read` is the test
        // that keeps it unreachable.
        Ok(u16::try_from(v).unwrap_or(u16::MAX))
    }

    fn put(&self, v: u16, shift: u32, bits: u32, invert: bool, dim: u16) -> u32 {
        let m = Self::field_mask(shift, bits) >> shift;
        let raw = if invert { Self::pivot(m, dim).saturating_sub(u32::from(v)) } else { u32::from(v) };
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
    /// The magic the first header line must start with — **version digit included**.
    ///
    /// `AEDAT` 3.1 and 4.0 are the same vendor, the same `.aedat` extension and completely
    /// different record layouts, so a prefix test of `"!AER-DAT"` alone admits both: a jAER
    /// `AEDAT` 3.1 file read this way decodes as a few hundred well-formed, entirely wrong
    /// `AEDAT` 2.0 events. The digit is the only thing on the wire that separates them, so it is
    /// part of the magic. See `a_missing_magic_stops_one_format_being_read_as_another`.
    pub const MAGIC: &'static str = "!AER-DAT2";
    /// Largest **first** timestamp an encodable stream may have, microseconds: 2^32 - 1.
    ///
    /// The record holds 32 bits of microseconds and the decoder's unwrapping accumulator starts at
    /// zero, so a first timestamp at or past 2^32 comes back reduced modulo 2^32 and nothing on the
    /// wire says how many wraps were dropped. Subtract the recording's start time before encoding.
    pub const MAX_TIME_US: u64 = (1u64 << 32) - 1;
    /// Largest gap between consecutive events that is always encodable, microseconds: 2^31 - 1.
    ///
    /// [`Aedat2::decode`] counts a wrap when the raw counter jumps backwards by **more than half**
    /// its range, which is the rule that stops one jittered record adding 71.6 minutes to the rest
    /// of a recording. The price of that rule is this bound: a gap of 2^31 or more that happens to
    /// straddle a wrap is indistinguishable from a backwards step, and a gap of 2^32 or more is
    /// not on the wire at all. Both are [`EncodeError::GapTooLarge`] rather than a file that
    /// decodes to a different recording — or, for a gap in `[2^31, 2^32)`, a file this crate's own
    /// decoder would refuse as non-monotonic.
    pub const MAX_GAP_US: u64 = (1u64 << 31) - 1;

    /// Decode a whole `AEDAT` 2.0 file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::LayoutFieldsOverlap`] if `layout` is self-contradictory;
    /// [`DecodeError::BadMagic`] if the first line is not `#!AER-DAT...`, which is what catches a
    /// file of another format being fed in; [`DecodeError::Truncated`] if the header has no
    /// terminator or the record array ends mid-record; [`DecodeError::FieldOutOfRange`] if a
    /// coordinate field names a column or row outside the layout's stated sensor geometry — the
    /// value reported is the field as it sits on the wire, before any mirroring — or if the layout
    /// mirrors a field about a width or height the field cannot hold;
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
                // A trigger only when the trigger bits are the only special bits set: on a DAVIS,
                // bit 31 makes bits 11-10 an APS or IMU sub-type, and bit 10 then is not a trigger.
                let kind = if addr & layout.special_mask & !layout.trigger_mask == 0 {
                    MarkerKind::ExternalTrigger
                } else {
                    MarkerKind::Other
                };
                markers.push(Marker { offset: at, kind, raw: u64::from(addr), t });
                at += 8;
                continue;
            }

            // A column or row past the stated sensor is reported as the field on the wire, since
            // a mirrored field past the sensor has no coordinate to report.
            let past = |field: &'static str, (raw, last): (u32, u32)| DecodeError::FieldOutOfRange {
                offset: at,
                field,
                value: u64::from(raw),
                max: u64::from(last),
            };
            let x = layout
                .get(addr, layout.x_shift, layout.x_bits, layout.x_invert, layout.width)
                .map_err(|e| past("column", e))?;
            let y = layout
                .get(addr, layout.y_shift, layout.y_bits, layout.y_invert, layout.height)
                .map_err(|e| past("row", e))?;
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
    /// Timestamps are written modulo 2^32 microseconds, which is what the format holds, so a
    /// recording longer than 71.6 minutes relies on the decoder's unwrapping to come back intact.
    /// That unwrapping recovers a wrap and **only** a wrap: it needs the first timestamp below
    /// 2^32 ([`Aedat2::MAX_TIME_US`]) and every gap below 2^31 ([`Aedat2::MAX_GAP_US`]), and this
    /// encoder refuses anything else rather than writing a file that decodes to a different
    /// recording. `aedat2_unwraps_a_timestamp_across_the_32_bit_wrap` checks the wrap it does
    /// recover and `aedat2_refuses_a_time_its_own_decoder_could_not_recover` checks the refusals.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order;
    /// [`EncodeError::FieldOutOfRange`] if a coordinate does not fit the layout's field width or
    /// its sensor geometry, if the layout mirrors a field about a width or height the field cannot
    /// hold, or if the first timestamp is past [`Aedat2::MAX_TIME_US`];
    /// [`EncodeError::GapTooLarge`] if two consecutive events are more than [`Aedat2::MAX_GAP_US`]
    /// apart; [`EncodeError::HeaderLineContainsNewline`] if a header line would not survive the
    /// round trip.
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
        let (x_span, y_span) = layout.coord_span();
        for (span, field) in [(x_span, "x_bits"), (y_span, "y_bits")] {
            if u64::from(span) > u64::from(u16::MAX) {
                return Err(EncodeError::FieldOutOfRange {
                    index: 0,
                    field,
                    value: u64::from(span),
                    max: u64::from(u16::MAX),
                });
            }
        }
        if let Some((field, value, max)) = layout.mirror_fault() {
            return Err(EncodeError::FieldOutOfRange { index: 0, field, value, max });
        }
        check_sorted(events)?;
        check_wrapping_clock(events, Self::MAX_TIME_US, Self::MAX_GAP_US)?;
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
            let mut addr = layout.put(e.x, layout.x_shift, layout.x_bits, layout.x_invert, layout.width)
                | layout.put(e.y, layout.y_shift, layout.y_bits, layout.y_invert, layout.height);
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
/// 4 hours 46 minutes before it wraps. [`Evt2::encode`] refuses a timestamp past
/// [`Evt2::MAX_TIME_US`], so no stream this crate writes can reach that wrap; the decoder counts it
/// anyway for a stream it did not write, and the hand-built streams in
/// `evt2_counts_a_real_2_34_wrap_and_refuses_a_jittered_one` are the one that reaches the branch
/// and the one that must not. A backwards step of half the field or less is corruption, not a
/// wrap, and is [`DecodeError::NonMonotonicTimestamp`].
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
    /// guess how much time went with it, or if an `EVT_TIME_HIGH` steps the 28-bit high field
    /// backwards by half its range or less, which is corruption rather than the 2^34 wrap.
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
                        // The field is 28 bits, so a wrap is a backwards jump of nearly 2^28 and
                        // anything smaller is corruption. Demanding MORE THAN HALF the range is
                        // the same rule AEDAT 2.0 and .dat use, and it is here for the same
                        // reason: the alternative adds 2^34 us — 4 h 46 min — to the rest of the
                        // recording on the strength of one jittered word. Measured before this
                        // check existed: a TIME_HIGH stepping back by ONE, from 10 to 9, put the
                        // next event 17,179,869,120 us after its predecessor, with no error.
                        if high - h > 1u32 << 27 {
                            wraps += 1;
                        } else {
                            return Err(DecodeError::NonMonotonicTimestamp {
                                offset: at,
                                previous: base,
                                found: (wraps << 34) | (u64::from(h) << 6),
                            });
                        }
                    }
                    high = h;
                    base = (wraps << 34) | (u64::from(high) << 6);
                    // A repeat carrying the value already held is deliberately NOT compared
                    // against the last event's time. A sensor emits a TIME_HIGH every 64 us
                    // whether or not the value changed, so a repeat is normal and must not be
                    // refused; the base it sets is below the last event's time by up to 63 us by
                    // construction. Time going backwards is caught where it matters, on the next
                    // event.
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
    /// Largest first timestamp an encodable stream may have, microseconds: 2^24 - 1, **inclusive**.
    ///
    /// The wire carries 24 bits of time and this is all of them set, so it is the last value that
    /// encodes rather than the first that does not; [`Evt3::encode`] refuses `MAX_FIRST_TIME_US + 1`
    /// and accepts this.
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
                    // Bits 11-8 of a VECT_8 word belong to no field. They are masked off here
                    // rather than refused: this review did not locate a statement that a device
                    // leaves them zero, and refusing them would be a guess that rejects real
                    // recordings. The mask is read only by the loop below, which for a VECT_8
                    // visits k = 0..=7, so masking them off is unobservable either way -- the
                    // observable choice is the refusal, and it is deliberately not made.
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
    /// [`EncodeError::FieldOutOfRange`] if the first timestamp is **past**
    /// [`Evt3::MAX_FIRST_TIME_US`] — that value itself encodes, as
    /// `evt3_encodes_its_largest_first_timestamp_and_refuses_the_next` shows — or a coordinate
    /// exceeds 2047; [`EncodeError::GapTooLarge`] if
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
/// strength of one record. [`Dat::encode`] refuses, symmetrically, the two things that unwrapping
/// cannot recover: a first timestamp past [`Dat::MAX_TIME_US`] and a gap past [`Dat::MAX_GAP_US`].
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
    /// Largest **first** timestamp an encodable stream may have, microseconds: 2^32 - 1.
    ///
    /// Same 32-bit wrapping clock as [`Aedat2::MAX_TIME_US`], and the same reason: the decoder's
    /// accumulator starts at zero, so the dropped high bits are nowhere on the wire.
    pub const MAX_TIME_US: u64 = (1u64 << 32) - 1;
    /// Largest gap between consecutive events that is always encodable, microseconds: 2^31 - 1.
    ///
    /// Same rule and same bound as [`Aedat2::MAX_GAP_US`]; the two formats share the clock and
    /// share the unwrapping.
    pub const MAX_GAP_US: u64 = (1u64 << 31) - 1;

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
    /// `Height` line in `header` that the caller supplied, or the first timestamp is past
    /// [`Dat::MAX_TIME_US`]; [`EncodeError::GapTooLarge`] if two consecutive events are more than
    /// [`Dat::MAX_GAP_US`] apart, which is the gap this format's 32-bit clock cannot be unwrapped
    /// back to; [`EncodeError::HeaderLineContainsNewline`] if a header line would not survive the
    /// round trip.
    pub fn encode(
        events: &[AerEvent],
        header: &[String],
        record_type: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        check_sorted(events)?;
        check_wrapping_clock(events, Self::MAX_TIME_US, Self::MAX_GAP_US)?;
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

    /// Read one of the two declared dimensions, refusing a value no `u16` coordinate could reach.
    fn dimension(bytes: &[u8], at: usize, field: &'static str) -> Result<u16, DecodeError> {
        let v = u32_le(bytes, at)?;
        u16::try_from(v).map_err(|_| DecodeError::FieldOutOfRange {
            offset: at,
            field,
            value: u64::from(v),
            max: u64::from(u16::MAX),
        })
    }

    /// Decode a flat file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::BadMagic`] if the first eight bytes are not [`Flat::MAGIC`];
    /// [`DecodeError::Truncated`] if the header is short; [`DecodeError::CountMismatch`] if the
    /// declared count and the bytes present disagree — in either direction, because trailing bytes
    /// after the last record mean the file is not what its header says it is;
    /// [`DecodeError::FieldOutOfRange`] for a polarity byte that is neither 0 nor 1, a declared
    /// width or height past 65535, or a coordinate past a declared width or height;
    /// [`DecodeError::ReservedNotZero`] for a dirty reserved byte;
    /// [`DecodeError::NonMonotonicTimestamp`] if the events are out of order.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let head = slice_at(bytes, 0, Self::HEADER_SIZE)?;
        if head[..8] != Self::MAGIC {
            return Err(DecodeError::BadMagic {
                offset: 0,
                expected: "FMAER-01",
                found: String::from_utf8_lossy(&head[..8]).into_owned(),
            });
        }
        // Refused, not saturated. The declared geometry is what every column and row below is
        // range-checked against, so a width of 100,000 folded to 65,535 would range-check every
        // record against a number that is in the file nowhere and reject or accept on it.
        let width = Self::dimension(bytes, 8, "width")?;
        let height = Self::dimension(bytes, 12, "height")?;
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

/// How an `AEDAT` 4.0 file's packets are compressed: field 0 of its `IOHeader`.
///
/// The field is `compression: CompressionType = NONE` in `dv-processing`'s `IOHeader.fbs`, with
/// `enum CompressionType : int32 { NONE, LZ4, LZ4_HIGH, ZSTD, ZSTD_HIGH }`, and iniVation's `AEDAT`
/// 4.0 page lists the same five: "Compression algorithm applied to all data streams in the file.
/// Currently supported are: NONE, LZ4, `LZ4_HIGH`, ZSTD and `ZSTD_HIGH`."
///
/// Through 0.22.0 this enum was read from a `Format:` text header line holding `RAW`,
/// `COMPRESSED_LZ4` and the like. `AEDAT` 4.0 has no such line and no such names, and a real `LZ4`
/// file that had got past the header would have been read as uncompressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aedat4Compression {
    /// `NONE`, 0, and the default when the field is absent: every packet is stored as it is. The
    /// only value whose events this crate can extract.
    None,
    /// `LZ4`, 1: each packet is one `LZ4` frame. The one `DV` recording this module was checked
    /// against uses it.
    Lz4,
    /// `LZ4_HIGH`, 2.
    Lz4High,
    /// `ZSTD`, 3: each packet is one `Zstandard` frame.
    Zstd,
    /// `ZSTD_HIGH`, 4.
    ZstdHigh,
    /// A value the schema does not define, kept so that an error can name it.
    Other(i32),
}

impl Aedat4Compression {
    /// The value as the `IOHeader` stores it.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Lz4High => 2,
            Self::Zstd => 3,
            Self::ZstdHigh => 4,
            Self::Other(code) => code,
        }
    }

    /// The name the schema gives this value, or `None` for a value it does not define.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        match self {
            Self::None => Some("NONE"),
            Self::Lz4 => Some("LZ4"),
            Self::Lz4High => Some("LZ4_HIGH"),
            Self::Zstd => Some("ZSTD"),
            Self::ZstdHigh => Some("ZSTD_HIGH"),
            Self::Other(_) => None,
        }
    }

    fn from_code(code: i32) -> Self {
        match code {
            0 => Self::None,
            1 => Self::Lz4,
            2 => Self::Lz4High,
            3 => Self::Zstd,
            4 => Self::ZstdHigh,
            other => Self::Other(other),
        }
    }

    /// The name, or the number for a value the schema does not define: what an error reports.
    fn label(self) -> String {
        self.name().map_or_else(|| format!("compression type {}", self.code()), str::to_string)
    }
}

/// One `AEDAT` 4.0 packet: an 8-byte header and a payload.
///
/// # The payload is the packet; the events are a view of it
///
/// [`Aedat4::encode`] writes [`Aedat4Packet::payload`] **verbatim**, and never the bytes it would
/// have produced from [`Aedat4Packet::events`]. That is what makes a re-save byte-exact for a file
/// this crate only partly understands: a compressed payload it cannot open, and an uncompressed one
/// laid out differently from the table [`Aedat4Packet::from_events`] writes — `FlatBuffers` is not
/// a canonical encoding, and a conforming writer may put its padding elsewhere. Regenerating the
/// payload from the decoded events would discard both.
///
/// This note used to add that "a real `dv-processing` events table carries fields beyond the
/// element vector". It does not. The schema, `include/dv-processing/data/event.fbs` in
/// `dv-processing` (gitlab.com/inivation/dv/dv-processing, last changed in commit c23aa373 of
/// 2024-09-18), declares `table EventPacket { elements: [Event] (native_inline); }` and nothing
/// else, and the event packets of the `DV` recording this module was checked against are laid out
/// byte for byte as this module's own writer lays them out.
///
/// The consequence for a caller who wants to *change* the events is that editing
/// [`Aedat4Packet::events`] alone changes nothing on disk. Use [`Aedat4Packet::set_events`], which
/// rewrites both, or [`Aedat4Packet::from_events`] to build a packet from scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat4Packet {
    /// The stream this packet belongs to, matching a node of the `infoNode` in the file's
    /// `IOHeader`. A recording interleaves streams — events, frames, inertial samples, triggers —
    /// and the id is the only thing in the packet header that separates them.
    pub stream_id: i32,
    /// Byte offset of this packet's 8-byte header in the whole file.
    pub offset: usize,
    /// The payload bytes exactly as they appeared, and exactly what a re-encode writes.
    ///
    /// Authoritative: see the note on [`Aedat4Packet`]. Not interpreted for a compressed file, and
    /// not regenerated for an uncompressed one.
    pub payload: Vec<u8>,
    /// Events, when the file is uncompressed and the payload is an event packet this
    /// implementation could walk; `None` when it was compressed or was not — a `DV` recording
    /// interleaves frames, inertial samples and triggers with its events, and those are packets in
    /// exactly the same container.
    pub events: Option<Vec<AerEvent>>,
    /// Why [`Aedat4Packet::events`] is `None` in an uncompressed file: the error that walking the
    /// payload produced. For a packet of another stream that is [`DecodeError::BadMagic`] naming
    /// the payload's `FlatBuffers` identifier — `FRME`, `IMUS`, `TRIG` — where `EVTS` was expected.
    ///
    /// `None` when the events decoded, and `None` for a compressed file, where no attempt was made
    /// and the reason is the `IOHeader`'s compression instead. [`Aedat4::events`] returns this
    /// error rather than inventing one, so a malformed event payload is still refused — it is the
    /// *framing* that survives it, which is what lets a caller reach the packets of a file whose
    /// other streams this crate cannot read.
    pub payload_error: Option<DecodeError>,
}

impl Aedat4Packet {
    /// Build an uncompressed event packet from events, payload and view consistent.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order, or
    /// [`EncodeError::FieldOutOfRange`] if a coordinate exceeds 32767 or a timestamp exceeds
    /// `i64::MAX`, the signed fields the `dv-processing` schema uses.
    pub fn from_events(stream_id: i32, events: &[AerEvent]) -> Result<Self, EncodeError> {
        let mut p = Self {
            stream_id,
            offset: 0,
            payload: Vec::new(),
            events: None,
            payload_error: None,
        };
        p.set_events(events)?;
        Ok(p)
    }

    /// Replace the events **and** the payload bytes that will be written for them.
    ///
    /// The only supported way to change what a packet contains: see the note on [`Aedat4Packet`]
    /// for why assigning to [`Aedat4Packet::events`] alone does not.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsorted`] if the events are not in timestamp order, or
    /// [`EncodeError::FieldOutOfRange`] if a coordinate exceeds 32767 or a timestamp exceeds
    /// `i64::MAX`.
    pub fn set_events(&mut self, events: &[AerEvent]) -> Result<(), EncodeError> {
        check_sorted(events)?;
        for (i, e) in events.iter().enumerate() {
            fit(i, "column", u64::from(e.x), 32767)?;
            fit(i, "row", u64::from(e.y), 32767)?;
            fit(i, "timestamp", e.t, u64::try_from(i64::MAX).unwrap_or(u64::MAX))?;
        }
        self.payload = Aedat4::write_event_packet(events);
        self.events = Some(events.to_vec());
        self.payload_error = None;
        Ok(())
    }
}

/// A bounds-checked view of one `FlatBuffers` buffer inside an `AEDAT` 4.0 file.
///
/// # Totality on a 32-bit target, which `wasm32` is
///
/// Every offset inside the buffer is a `u32` the file chose, and this crate compiles to `wasm32`,
/// where `usize` is 32 bits. Adding those `u32`s into a `usize` overflows there — a panic under
/// debug overflow checks, a wrap in release — and a wrap is the dangerous half: a root offset of
/// `0xFFFF_FFFE` plus four becomes 2, which passes every bounds check that follows. So the
/// arithmetic is done in `u64` and narrowed to `usize` only once the value is known to lie inside
/// the buffer, which makes the 32-bit and the 64-bit path the same path and lets
/// `aedat4_refuses_offsets_that_would_wrap_a_32_bit_usize` prove it on a 64-bit host.
struct Fb<'a> {
    p: &'a [u8],
    /// Where `p` starts in the whole file, for the offset an error reports.
    base: usize,
}

impl Fb<'_> {
    fn bad(&self, what: &'static str) -> DecodeError {
        DecodeError::MalformedFlatBuffer { offset: self.base, what }
    }

    /// `at` as a `usize`, once `at .. at + n` is known to lie wholly inside the buffer, which also
    /// makes the narrowing exact.
    fn spot(&self, at: u64, n: u64, what: &'static str) -> Result<usize, DecodeError> {
        let len = self.p.len() as u64;
        let end = at.checked_add(n).filter(|&e| e <= len).ok_or(self.bad(what))?;
        debug_assert!(end <= len);
        usize::try_from(at).map_err(|_| self.bad(what))
    }

    fn u32_at(&self, at: u64, what: &'static str) -> Result<u32, DecodeError> {
        u32_le(self.p, self.spot(at, 4, what)?).map_err(|_| self.bad(what))
    }

    /// Refuse unless the four-character file identifier at `at` is `want`.
    fn identifier(&self, at: u64, want: &'static str) -> Result<(), DecodeError> {
        let i = self.spot(at, 4, "file identifier")?;
        let found = &self.p[i..i + 4];
        if found != want.as_bytes() {
            return Err(DecodeError::BadMagic {
                offset: self.base.saturating_add(i),
                expected: want,
                found: String::from_utf8_lossy(found).into_owned(),
            });
        }
        Ok(())
    }

    /// Where field `index` of the table at `table` lies, found through the table's vtable — or
    /// `None` when the vtable does not record the field, which `FlatBuffers` means as "the schema's
    /// default".
    fn field(&self, table: u64, index: u64) -> Result<Option<u64>, DecodeError> {
        let soffset = self.u32_at(table, "root offset")? as i32;
        let vtable = i64::try_from(table).map_err(|_| self.bad("vtable"))? - i64::from(soffset);
        let vtable = u64::try_from(vtable).map_err(|_| self.bad("vtable"))?;
        let vt_at = self.spot(vtable, 4, "vtable")?;
        let vt_len = u64::from(u16_le(self.p, vt_at).map_err(|_| self.bad("vtable"))?);
        self.spot(vtable, vt_len, "vtable")?;
        let slot = 4 + 2 * index;
        if vt_len < slot + 2 {
            // A vtable that stops short of the slot has not recorded the field.
            return Ok(None);
        }
        let entry = self.spot(vtable + slot, 2, "vtable")?;
        let within = u64::from(u16_le(self.p, entry).map_err(|_| self.bad("vtable"))?);
        if within == 0 {
            return Ok(None);
        }
        Ok(Some(table + within))
    }
}

/// A decoded `AEDAT` 4.0 file — **framing complete, payload decoding partial, and this doc says
/// exactly where the line is**.
///
/// The container, as iniVation's `AEDAT` 4.0 page and `dv-processing`'s reader lay it out:
///
/// 1. the 14-byte version line [`Aedat4::MAGIC`];
/// 2. a 4-byte little-endian size and that many bytes of `IOHeader`, a `FlatBuffers` table with
///    the file identifier `IOHE` whose fields are the compression, the position of the file data
///    table, and an XML `infoNode` describing the streams;
/// 3. packets, each an 8-byte header — little-endian `int32` stream id, little-endian `int32`
///    payload size — followed by the payload, which once decompressed "is a size-prefixed
///    Flatbuffer" carrying its own four-character identifier;
/// 4. when the `IOHeader`'s `dataTablePosition` is not -1, a `FileDataTable` from that offset to
///    the end: "No more data is present after that offset in the file."
///
/// [`Aedat4::decode`] reads all of that, for every file, whatever the compression: you always get
/// the stream ids, the packet boundaries and the byte offsets.
///
/// # What this module used to get wrong
///
/// Through 0.22.0 it read `#`-prefixed text header lines after the version line, up to a
/// `#!END-HEADER` line, and took the compression from a `Format:` line among them; it read a
/// payload's `FlatBuffers` root offset from byte 0; and it read packets to the end of the file.
/// None of that is the format. There is no text header: byte 14 of a `DV` recording is the first
/// byte of the `IOHeader` size — `0x24` in the recording below, not `#` — so every file `DV`
/// writes was refused there with [`DecodeError::Truncated`]. The compression is the `IOHeader`'s
/// first field. A payload's first four bytes are its size prefix, which the old reader took for
/// the root offset. And the data table's bytes are not packets. The round-trip tests passed
/// throughout, because this module's writer made the same mistakes its reader did — which is what
/// an encoder used as its own decoder's oracle cannot see.
///
/// # Checked against a recording `DV` wrote
///
/// The `neuromorphicsystems/aedat` reader's `tests/data/test_data.aedat4`, recorded from
/// `DAVIS346_00000002` with `LZ4` compression: 6,144,209 bytes, a 3,108-byte `IOHeader`, 708
/// packets in four streams (events, frames, inertial samples, triggers) and a 22,166-byte data
/// table at byte 6,122,043. [`Aedat4::decode`] reads that framing and [`Aedat4::encode`] writes the
/// file back byte for byte. With every packet and the table decompressed outside this crate
/// (`Python`'s `lz4`) and the `IOHeader` patched to `NONE`, the 236 event packets decode to the
/// same 78,830 events as an independent `Python` walker, and the other 472 packets are refused by
/// their identifiers. Rebuilt from its own decoded events by [`Aedat4Packet::from_events`], every
/// one of the 236 event packets comes out byte for byte as `DV` wrote it;
/// `aedat4_reads_and_writes_the_event_table_dv_wrote` pins that layout on the recording's first
/// two events.
///
/// # What this implementation cannot do
///
/// **Compressed payloads.** Each packet of an `LZ4` or `Zstd` file is one compressed frame. This
/// crate has zero dependencies and carries no decompressor, so a compressed packet arrives with
/// `events: None` and [`Aedat4::events`] returns [`DecodeError::UnsupportedCompression`] naming
/// the compression and the offset. That is a capability boundary of a zero-dependency crate,
/// stated rather than hidden; the one `DV` recording checked is `LZ4`, so decompress upstream.
///
/// **Streams that are not events.** A `DV` recording interleaves frames, inertial samples and
/// triggers with its events, as packets in this same container. Those arrive with `events: None`
/// and a [`DecodeError::BadMagic`] naming their identifier in [`Aedat4Packet::payload_error`]; the
/// framing — stream id, offset, payload — is complete for them, which is what lets a caller pick
/// out the event stream and hand the rest to a `FlatBuffers` reader. [`Aedat4::events`] is a
/// whole-file call and refuses if any packet is not events, because a silently short event list is
/// the worse answer; to read one stream, filter [`Aedat4::packets`] by its id.
///
/// **The stream description.** The `IOHeader`'s `infoNode` is kept verbatim in
/// [`Aedat4::io_header`] and not parsed, so this crate knows streams only by their ids, and a file
/// it writes carries no description at all. Assume such a file does not open in `DV` until
/// someone checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat4 {
    /// The compression the `IOHeader` declared when the file was decoded.
    ///
    /// A view, like [`Aedat4Packet::events`]: [`Aedat4::encode`] writes [`Aedat4::io_header`],
    /// and changing this field changes nothing on disk.
    pub compression: Aedat4Compression,
    /// The `IOHeader` verbatim, without the 4-byte size in front of it. Authoritative: it is what a
    /// re-encode writes, and the compression and the data table's position are read from it.
    pub io_header: Vec<u8>,
    /// Packets in file order.
    pub packets: Vec<Aedat4Packet>,
    /// The `FileDataTable` verbatim, from the `IOHeader`'s `dataTablePosition` to the end of the
    /// file, still compressed if the file is; empty when the position is -1. Not interpreted.
    pub data_table: Vec<u8>,
}

impl Aedat4 {
    /// The version line every file starts with: exactly 14 bytes, `dv-processing`'s
    /// `AEDAT_VERSION_LENGTH`.
    ///
    /// This constant used to be `"!AER-DAT4"`, the prefix of a `#`-prefixed header line, and a
    /// sibling `END_HEADER` named the line that closed the header. There is no such header and no
    /// such line; see the note on [`Aedat4`].
    pub const MAGIC: &'static str = "#!AER-DAT4.0\r\n";
    /// The `FlatBuffers` file identifier of an `IOHeader`.
    pub const IO_HEADER_ID: &'static str = "IOHE";
    /// The `FlatBuffers` file identifier of an event packet, `file_identifier "EVTS"` in
    /// `dv-processing`'s `event.fbs`.
    pub const EVENTS_ID: &'static str = "EVTS";
    /// Bytes per event inside an uncompressed payload's `FlatBuffers` struct vector.
    pub const FB_EVENT_SIZE: usize = 16;

    /// The smallest `IOHeader` the schema admits: compression `NONE`, written out rather than left
    /// to the default, no `dataTablePosition` (so -1) and no `infoNode`.
    ///
    /// ```text
    ///  0.. 4  uint32 root offset = 16
    ///  4.. 8  file identifier "IOHE"
    ///  8..10  padding
    /// 10..12  uint16 vtable length = 6
    /// 12..14  uint16 table length = 8
    /// 14..16  uint16 offset of field 0 (compression) within the table = 4
    /// 16..20  int32 soffset back to the vtable = 6
    /// 20..24  int32 compression = 0, NONE
    /// ```
    const MINIMAL_IO_HEADER: [u8; 24] = [
        16, 0, 0, 0, b'I', b'O', b'H', b'E', 0, 0, 6, 0, 8, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// Decode the container, and the payloads of an uncompressed file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::BadMagic`] at the first byte that differs from [`Aedat4::MAGIC`], or if the
    /// `IOHeader` does not carry the identifier `IOHE`; [`DecodeError::Truncated`] if the file ends
    /// inside the version line, or a size field or payload runs off the end of the file or into the
    /// data table; [`DecodeError::FieldOutOfRange`] for a negative size field;
    /// [`DecodeError::MalformedFlatBuffer`] for an `IOHeader` whose offsets point outside it, or
    /// whose `dataTablePosition` is neither -1 nor a position between the end of the `IOHeader` and
    /// the end of the file. The `IOHeader` is refused rather than defaulted because without it the
    /// compression is unknown, and a guess would read compressed bytes as events.
    ///
    /// A payload this implementation cannot walk, **or walks and rejects** — a frame or an inertial
    /// stream, a malformed buffer, a declared count the bytes cannot supply, a negative or backwards
    /// timestamp — does not fail the decode: the framing is what this call promises for every file,
    /// so the packet comes back with `events: None` and the reason in
    /// [`Aedat4Packet::payload_error`], and [`Aedat4::events`] is where it is refused.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let magic = Self::MAGIC.as_bytes();
        if let Some(i) = magic.iter().zip(bytes).position(|(m, b)| m != b) {
            return Err(DecodeError::BadMagic {
                offset: i,
                expected: Self::MAGIC,
                found: String::from_utf8_lossy(&bytes[..bytes.len().min(magic.len())]).into_owned(),
            });
        }
        // Every byte present matched, so a file shorter than the line is a cut-short one.
        slice_at(bytes, 0, magic.len())?;
        let mut at = magic.len();
        let io_len = Self::signed_len(bytes, at, "IOHeader size")?;
        at += 4;
        let io_header = slice_at(bytes, at, io_len)?.to_vec();
        let (compression, table_at) = Self::read_io_header(&io_header, at)?;
        at += io_len;

        // "No more data is present after that offset in the file": the packets stop where the
        // data table starts, and its bytes are not packets. Read as a packet header, the first
        // eight bytes of the recording's LZ4-compressed table are a negative size.
        let end = match table_at {
            None => bytes.len(),
            Some(p) => usize::try_from(p).ok().filter(|p| (at..=bytes.len()).contains(p)).ok_or(
                DecodeError::MalformedFlatBuffer { offset: magic.len() + 4, what: "dataTablePosition" },
            )?,
        };
        let region = &bytes[..end];

        let mut packets = Vec::new();
        // Carried ACROSS packets: a file whose packet boundaries each hold sorted events but whose
        // packets are out of order would otherwise pass every per-packet check and still hand the
        // caller a non-monotonic stream. A mutation fuzzer found exactly that.
        let mut last_t = 0u64;
        while at < end {
            let head = at;
            let stream_id = u32_le(region, at)? as i32;
            let size = Self::signed_len(region, at + 4, "packet size")?;
            let payload = slice_at(region, at + 8, size)?.to_vec();
            // A file interleaves streams: DV writes frames, inertial samples and triggers as
            // packets in this same container, and they are not event tables. Applying the event
            // reader to every packet AND propagating its error refused the whole file — so no
            // real DV recording opened at all, which is the opposite of what this type promises.
            // The error is kept on the packet instead and returned by `events()`, so the refusal
            // survives without taking the framing with it.
            let (events, payload_error) = if compression == Aedat4Compression::None {
                match Self::read_event_packet(&payload, at + 8, &mut last_t) {
                    Ok(e) => (Some(e), None),
                    Err(err) => (None, Some(err)),
                }
            } else {
                (None, None)
            };
            packets.push(Aedat4Packet { stream_id, offset: head, payload, events, payload_error });
            at += 8 + size;
        }
        Ok(Self { compression, io_header, packets, data_table: bytes[end..].to_vec() })
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

    /// The two `IOHeader` fields this implementation acts on: `compression`, field 0, default
    /// `NONE`; and `dataTablePosition`, field 1, default -1, returned as `None`.
    ///
    /// `base` is where the `IOHeader` starts in the file. The container's size field in front of
    /// it is the `FlatBuffer`'s size prefix, so here the root offset is at byte 0 and the
    /// identifier at byte 4.
    fn read_io_header(h: &[u8], base: usize) -> Result<(Aedat4Compression, Option<u64>), DecodeError> {
        let fb = Fb { p: h, base };
        fb.identifier(4, Self::IO_HEADER_ID)?;
        let root = u64::from(fb.u32_at(0, "root offset")?);
        let compression = match fb.field(root, 0)? {
            Some(at) => Aedat4Compression::from_code(fb.u32_at(at, "compression")? as i32),
            None => Aedat4Compression::None,
        };
        let position = match fb.field(root, 1)? {
            Some(at) => {
                let lo = u64::from(fb.u32_at(at, "dataTablePosition")?);
                let hi = u64::from(fb.u32_at(at + 4, "dataTablePosition")?);
                (lo | (hi << 32)) as i64
            }
            None => -1,
        };
        if position == -1 {
            return Ok((compression, None));
        }
        let position = u64::try_from(position).map_err(|_| fb.bad("dataTablePosition"))?;
        Ok((compression, Some(position)))
    }

    /// Walk an uncompressed event packet: a size-prefixed `FlatBuffer` with the identifier `EVTS`.
    /// Every offset is bounds-checked before use, in `u64`; see [`Fb`].
    ///
    /// The size prefix counts the bytes after it, and the root offset that follows counts from
    /// byte 4, not from byte 0: "All Flatbuffers are size-prefixed, meaning the first four bytes
    /// represent a 32 bit little-endian integer encoding the size of the following, actual
    /// Flatbuffer data." Through 0.22.0 this reader took the size prefix for the root offset and its
    /// writer wrote no prefix, so the two agreed with each other and with no file `DV` writes.
    ///
    /// `last_t` is advanced only if the whole packet decodes, so a packet that is refused does not
    /// leave the cross-packet monotonicity cursor somewhere in the middle of itself.
    fn read_event_packet(
        p: &[u8],
        base: usize,
        last_t: &mut u64,
    ) -> Result<Vec<AerEvent>, DecodeError> {
        let fb = Fb { p, base };
        let len = p.len() as u64;
        let prefix = u64::from(fb.u32_at(0, "size prefix")?);
        if prefix != len - 4 {
            return Err(DecodeError::CountMismatch { offset: base, declared: prefix, actual: len - 4 });
        }
        fb.identifier(8, Self::EVENTS_ID)?;
        let root = 4 + u64::from(fb.u32_at(4, "root offset")?);
        let Some(slot) = fb.field(root, 0)? else {
            // The table records no element vector: an empty packet.
            return Ok(Vec::new());
        };
        let vec_at = slot + u64::from(fb.u32_at(slot, "vector")?);
        let vec_head = fb.spot(vec_at, 4, "vector")?;
        let count = u64::from(u32_le(p, vec_head).map_err(|_| fb.bad("vector"))?);
        let size = Self::FB_EVENT_SIZE as u64;
        let first = vec_at + 4;
        let bytes_needed = count.checked_mul(size).ok_or_else(|| fb.bad("vector"))?;
        if first.checked_add(bytes_needed).is_none_or(|end| end > len) {
            return Err(DecodeError::CountMismatch {
                offset: base.saturating_add(vec_head),
                declared: count,
                actual: (len - first) / size,
            });
        }
        let mut events = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        // Committed to `*last_t` only on success; see the note above.
        let mut cursor = *last_t;
        for k in 0..count {
            let at = fb.spot(first + k * size, size, "vector")?;
            let raw_t = u64_le(p, at).map_err(|_| fb.bad("vector"))? as i64;
            let t = u64::try_from(raw_t).map_err(|_| DecodeError::FieldOutOfRange {
                offset: base.saturating_add(at),
                field: "timestamp",
                value: raw_t.unsigned_abs(),
                max: u64::try_from(i64::MAX).unwrap_or(u64::MAX),
            })?;
            if t < cursor {
                return Err(DecodeError::NonMonotonicTimestamp {
                    offset: base.saturating_add(at),
                    previous: cursor,
                    found: t,
                });
            }
            cursor = t;
            let sx = u16_le(p, at + 8).map_err(|_| fb.bad("vector"))? as i16;
            let sy = u16_le(p, at + 10).map_err(|_| fb.bad("vector"))? as i16;
            for (v, field) in [(sx, "column"), (sy, "row")] {
                if v < 0 {
                    return Err(DecodeError::FieldOutOfRange {
                        offset: base.saturating_add(at + 8),
                        field,
                        value: u64::from(v.unsigned_abs()),
                        max: u64::from(i16::MAX.unsigned_abs()),
                    });
                }
            }
            let pol_byte = *p.get(at + 12).ok_or_else(|| fb.bad("vector"))?;
            let polarity = match pol_byte {
                0 => Polarity::Off,
                1 => Polarity::On,
                other => {
                    return Err(DecodeError::FieldOutOfRange {
                        offset: base.saturating_add(at + 12),
                        field: "polarity",
                        value: u64::from(other),
                        max: 1,
                    });
                }
            };
            events.push(AerEvent { t, x: sx.unsigned_abs(), y: sy.unsigned_abs(), polarity });
        }
        *last_t = cursor;
        Ok(events)
    }

    /// All events from every packet, in file order.
    ///
    /// # Errors
    ///
    /// [`DecodeError::UnsupportedCompression`] naming the packet's byte offset and the file's
    /// compression, if a packet's payload was compressed and so never read. This is where the
    /// crate's zero-dependency boundary becomes visible to a caller, and it refuses with the reason
    /// rather than returning a partial list that looks like a short recording.
    ///
    /// Otherwise, whatever [`Aedat4Packet::payload_error`] holds: a packet of an uncompressed file
    /// that is not an event table — a frame, an inertial sample or a trigger in an interleaved `DV`
    /// recording, or a malformed buffer — is refused here with the error that walking it produced,
    /// verbatim, so that "this stream is not events" and "this crate has no decompressor" cannot be
    /// confused.
    pub fn events(&self) -> Result<Vec<AerEvent>, DecodeError> {
        let mut all = Vec::new();
        for p in &self.packets {
            match (&p.events, &p.payload_error) {
                (Some(e), _) => all.extend_from_slice(e),
                (None, Some(err)) => return Err(err.clone()),
                (None, None) => {
                    return Err(DecodeError::UnsupportedCompression {
                        offset: p.offset,
                        name: self.compression.label(),
                    });
                }
            }
        }
        Ok(all)
    }

    /// Build an uncompressed `AEDAT` 4.0 file from events, one packet per `events_per_packet`.
    ///
    /// The `IOHeader` is the smallest the schema admits — compression `NONE`, no data table, no
    /// `infoNode` — because this implementation does not synthesise a stream description it cannot
    /// verify against `DV`. A file produced here therefore round-trips through [`Aedat4::decode`]
    /// exactly and should be assumed **not** to open in `DV` until someone checks.
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
            packets.push(Aedat4Packet::from_events(stream_id, part).map_err(|e| match e {
                // The index `set_events` reports is into the chunk; the caller indexed the whole
                // slice, and an error naming event 3 of 8,000 when it meant 2,051 is worse than
                // no index at all.
                EncodeError::FieldOutOfRange { index, field, value, max } => {
                    EncodeError::FieldOutOfRange { index: c * chunk + index, field, value, max }
                }
                EncodeError::Unsorted { index, previous, found } => {
                    EncodeError::Unsorted { index: c * chunk + index, previous, found }
                }
                other => other,
            })?);
        }
        Ok(Self {
            compression: Aedat4Compression::None,
            io_header: Self::MINIMAL_IO_HEADER.to_vec(),
            packets,
            data_table: Vec::new(),
        })
    }

    /// The event packet this implementation writes: the layout `DV` gave every event packet of the
    /// one recording this module was checked against, all 236 of which this function reproduces
    /// byte for byte from their own events.
    ///
    /// ```text
    ///  0.. 4  uint32 size prefix: the bytes after it
    ///  4.. 8  uint32 root offset, counted from byte 4 = 16, so the table is at 20
    ///  8..12  file identifier "EVTS"
    /// 12..14  padding
    /// 14..16  uint16 vtable length = 6
    /// 16..18  uint16 table length = 8
    /// 18..20  uint16 offset of field 0, the element vector, within the table = 4
    /// 20..24  int32 soffset back to the vtable = 6
    /// 24..28  uint32 offset from this slot to the vector = 4
    /// 28..32  uint32 element count
    /// 32..    elements, 16 bytes each, 8-aligned as the int64 field requires
    /// ```
    fn write_event_packet(events: &[AerEvent]) -> Vec<u8> {
        let after_prefix = 28 + events.len() * Self::FB_EVENT_SIZE;
        let mut p = Vec::with_capacity(4 + after_prefix);
        p.extend_from_slice(&(after_prefix as u32).to_le_bytes());
        p.extend_from_slice(&16u32.to_le_bytes());
        p.extend_from_slice(Self::EVENTS_ID.as_bytes());
        p.extend_from_slice(&[0u8; 2]);
        p.extend_from_slice(&6u16.to_le_bytes());
        p.extend_from_slice(&8u16.to_le_bytes());
        p.extend_from_slice(&4u16.to_le_bytes());
        p.extend_from_slice(&6i32.to_le_bytes());
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
    /// Every packet is written from [`Aedat4Packet::payload`], byte for byte, whatever the
    /// compression and whether or not its events were decoded, and so are the `IOHeader` and the
    /// data table. So **any** file this crate decodes survives a decode-encode cycle unchanged — a
    /// compressed one it cannot read inside, and equally an uncompressed one whose tables are laid
    /// out differently from the ones this module writes. Regenerating a payload from its decoded
    /// events drops such differences silently:
    /// `aedat4_re_saves_a_raw_packet_it_did_not_lay_out_itself_byte_for_byte` builds a 56-byte
    /// packet with an 8-byte gap in it, which the writer would put back as 48.
    ///
    /// Changing what a packet holds therefore goes through [`Aedat4Packet::set_events`] rather than
    /// through the `events` field — see the note on [`Aedat4Packet`] — and in a file with a data
    /// table it then needs [`Aedat4::drop_data_table`], because the table indexes the packets by
    /// byte offset.
    ///
    /// # Errors
    ///
    /// [`EncodeError::FieldOutOfRange`] if the `IOHeader` or a payload is longer than the `int32`
    /// size field can express, which is a file that cannot be written rather than one written with
    /// a wrong length in it, or if [`Aedat4::io_header`] is not an `IOHeader` this crate can read,
    /// since then nothing says where the data table belongs; [`EncodeError::DataTableMisplaced`] if
    /// the `IOHeader`'s `dataTablePosition` is not where the packets end, or is -1 while there is a
    /// table to write. Coordinate ranges were enforced when the packets were built.
    pub fn encode(&self) -> Result<Vec<u8>, EncodeError> {
        let (_, declared) = Self::read_io_header(&self.io_header, Self::MAGIC.len() + 4).map_err(|_| {
            EncodeError::FieldOutOfRange {
                index: 0,
                field: "an IOHeader this crate cannot read",
                value: self.io_header.len() as u64,
                max: 0,
            }
        })?;
        let mut out = Vec::from(Self::MAGIC.as_bytes());
        out.extend_from_slice(&Self::size_field(0, "IOHeader size", self.io_header.len())?.to_le_bytes());
        out.extend_from_slice(&self.io_header);
        for (i, p) in self.packets.iter().enumerate() {
            let size = Self::size_field(i, "packet size", p.payload.len())?;
            out.extend_from_slice(&p.stream_id.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&p.payload);
        }
        let packets_end = out.len() as u64;
        let placed = match declared {
            Some(at) => at == packets_end,
            None => self.data_table.is_empty(),
        };
        if !placed {
            return Err(EncodeError::DataTableMisplaced { declared, packets_end });
        }
        out.extend_from_slice(&self.data_table);
        Ok(out)
    }

    /// Remove the file data table, and set the `IOHeader`'s `dataTablePosition` to -1 so that it
    /// says so.
    ///
    /// The table indexes the packets by byte offset, so it is stale the moment a packet changes
    /// size, and [`Aedat4::encode`] refuses to write one anywhere but where the `IOHeader` says it
    /// starts. Without it a reader walks the packets from the front — "'-1' means no table
    /// present" — which is all this crate does in any case.
    ///
    /// # Errors
    ///
    /// Whatever reading [`Aedat4::io_header`] produced, if it cannot be read: there is then no
    /// field to rewrite, and nothing is changed.
    pub fn drop_data_table(&mut self) -> Result<(), DecodeError> {
        let slot = {
            let fb = Fb { p: &self.io_header, base: Self::MAGIC.len() + 4 };
            Self::read_io_header(fb.p, fb.base)?;
            let root = u64::from(fb.u32_at(0, "root offset")?);
            fb.field(root, 1)?.map(|at| fb.spot(at, 8, "dataTablePosition")).transpose()?
        };
        if let Some(i) = slot {
            self.io_header[i..i + 8].copy_from_slice(&(-1i64).to_le_bytes());
        }
        self.data_table.clear();
        Ok(())
    }

    /// A length as the container's signed 32-bit size field, or a refusal.
    ///
    /// Saturating it at `i32::MAX` would write a **wrong number** into the file — a length field
    /// that disagrees with the bytes that follow it, which is the one thing a container's framing
    /// must never do. `a_payload_too_long_for_the_size_field_is_refused_not_truncated` calls this
    /// with a length no test can allocate.
    fn size_field(index: usize, field: &'static str, len: usize) -> Result<i32, EncodeError> {
        i32::try_from(len).map_err(|_| EncodeError::FieldOutOfRange {
            index,
            field,
            value: len as u64,
            max: i32::MAX.unsigned_abs().into(),
        })
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


// ---------------------------------------------------------------------------------------------
// N-MNIST / N-Caltech101 records
// ---------------------------------------------------------------------------------------------

/// The five-byte event record of the N-MNIST and N-Caltech101 datasets (Orchard, Jayawant, Cohen
/// and Thakor, *Converting static image datasets to spiking neuromorphic datasets using saccades*,
/// Frontiers in Neuroscience 9:437, 2015), as their distribution's README states it:
///
/// | bits | field |
/// |---|---|
/// | 39–32 | column `x`, pixels |
/// | 31–24 | row `y`, pixels |
/// | 23 | polarity, `1` for ON |
/// | 22–0 | timestamp, microseconds |
///
/// Big-endian within the record. A 23-bit microsecond timestamp wraps at 8.39 s; the dataset's
/// samples are a third of a second, and this decoder does **not** unwrap — a decrease is reported
/// as [`DecodeError::NonMonotonicTimestamp`], because a wrap and a corrupt record are the same
/// bytes and only the caller knows which it has.
///
/// The record carries no geometry. N-MNIST is 34 × 34 (an MNIST digit with a 3-pixel border after
/// the saccade); N-Caltech101 frames are up to 240 × 180. The coordinates are returned as read and
/// range-checked against nothing, and [`TrainMap`] is where a caller states the width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NMnist {
    /// Events in file order.
    pub events: Vec<AerEvent>,
}

impl NMnist {
    /// Bytes per record.
    pub const RECORD_SIZE: usize = 5;
    /// The largest timestamp the 23-bit field holds, microseconds.
    pub const MAX_TIMESTAMP: u64 = (1 << 23) - 1;

    /// Decode a file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] if the length is not a multiple of five — the partial record at
    /// the end is named by offset; [`DecodeError::NonMonotonicTimestamp`] if a timestamp is below
    /// its predecessor.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let n = bytes.len() / Self::RECORD_SIZE;
        let rem = bytes.len() % Self::RECORD_SIZE;
        if rem != 0 {
            return Err(DecodeError::Truncated { offset: n * Self::RECORD_SIZE, need: Self::RECORD_SIZE, have: rem });
        }
        let mut events = Vec::with_capacity(n);
        let mut previous: Option<u64> = None;
        for k in 0..n {
            let at = k * Self::RECORD_SIZE;
            let r = slice_at(bytes, at, Self::RECORD_SIZE)?;
            let x = u16::from(r[0]);
            let y = u16::from(r[1]);
            let polarity = if r[2] & 0x80 != 0 { Polarity::On } else { Polarity::Off };
            let t = (u64::from(r[2] & 0x7F) << 16) | (u64::from(r[3]) << 8) | u64::from(r[4]);
            if let Some(p) = previous.filter(|&p| t < p) {
                return Err(DecodeError::NonMonotonicTimestamp { offset: at, previous: p, found: t });
            }
            previous = Some(t);
            events.push(AerEvent { t, x, y, polarity });
        }
        Ok(Self { events })
    }

    /// Encode events in the same layout. Every coordinate must fit a byte and every timestamp the
    /// 23-bit field; the result decodes bit-exactly.
    ///
    /// Returns `None`, naming nothing, when a coordinate is past 255 or a timestamp past
    /// [`NMnist::MAX_TIMESTAMP`]: the format has no room for it and no way to say so.
    #[must_use]
    pub fn encode(events: &[AerEvent]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(events.len() * Self::RECORD_SIZE);
        for e in events {
            if e.x > 255 || e.y > 255 || e.t > Self::MAX_TIMESTAMP {
                return None;
            }
            let p: u8 = if e.polarity == Polarity::On { 0x80 } else { 0 };
            out.push(e.x as u8);
            out.push(e.y as u8);
            out.push(p | ((e.t >> 16) as u8 & 0x7F));
            out.push((e.t >> 8) as u8);
            out.push(e.t as u8);
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------------------------
// AEDAT 3.1 and the DVS128 Gesture dataset
// ---------------------------------------------------------------------------------------------

/// An `AEDAT` 3.1 file: the container of the DVS128 Gesture dataset (Amir, Taba, Berg, Melano,
/// `McKinstry`, Di Nolfo, Nayak, Andreopoulos, Garreau, Mendoza, Kusnitz, Debole, Esser, Delbruck,
/// Flickner and Modha, *A low power, fully event-based gesture recognition system*, CVPR 2017,
/// pp. 7388–7397).
///
/// Transcribed from the vendor's specification (iniVation, *`AEDAT` 3.1 file format*) and checked
/// against it field by field: a text header from `#!AER-DAT3.1\r\n` to `#!END-HEADER\r\n`, then
/// packets, each a 28-byte little-endian header — `eventType` (2 bytes), `eventSource` (2),
/// `eventSize` (4), `eventTSOffset` (4), `eventTSOverflow` (4), `eventCapacity` (4), `eventNumber`
/// (4), `eventValid` (4) — followed by `eventNumber` events of `eventSize` bytes. A POLARITY event
/// (type 1) is 8 bytes: a data word with validity in bit 0, polarity in bit 1, `y` in bits 2–16
/// and `x` in bits 17–31, then a 32-bit microsecond timestamp; the full time is
/// `(eventTSOverflow << 31) | timestamp`.
///
/// Packets of every other type are skipped by their declared length and counted. Events whose
/// validity bit is clear are dropped and counted. Nothing is reordered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aedat3 {
    /// Valid polarity events, in file order.
    pub events: Vec<AerEvent>,
    /// Polarity events dropped because their validity bit was clear.
    pub invalid: u64,
    /// Packets of other types (special, frame, IMU, …) that were skipped.
    pub skipped_packets: u64,
}

impl Aedat3 {
    /// The first line of every file.
    pub const MAGIC: &'static str = "#!AER-DAT3.1\r\n";
    /// The line that ends the text header.
    pub const END_HEADER: &'static str = "#!END-HEADER\r\n";
    /// Bytes in a packet header.
    pub const PACKET_HEADER: usize = 28;
    /// The `eventType` of a polarity packet.
    pub const POLARITY: u16 = 1;

    /// Decode a file.
    ///
    /// # Errors
    ///
    /// [`DecodeError::BadMagic`] if the file does not start with [`Aedat3::MAGIC`] or has no
    /// [`Aedat3::END_HEADER`]; [`DecodeError::Truncated`] for a packet header or body cut short;
    /// [`DecodeError::UnsupportedRecordLayout`] for a polarity packet whose events are not 8 bytes
    /// with the timestamp at offset 4; [`DecodeError::CountMismatch`] when `eventCapacity` differs
    /// from `eventNumber` (the specification says they are equal in files) or `eventValid` from
    /// the valid events found; [`DecodeError::FieldOutOfRange`] for a negative count, a timestamp
    /// with its sign bit set, or an overflow counter past what a `u64` of microseconds holds;
    /// [`DecodeError::NonMonotonicTimestamp`] if a polarity event is earlier than the one before.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let magic = Self::MAGIC.as_bytes();
        if bytes.len() < magic.len() || &bytes[..magic.len()] != magic {
            let found = String::from_utf8_lossy(&bytes[..bytes.len().min(magic.len())]).into_owned();
            return Err(DecodeError::BadMagic { offset: 0, expected: Self::MAGIC, found });
        }
        let end = Self::END_HEADER.as_bytes();
        let Some(header_end) = bytes.windows(end.len()).position(|w| w == end) else {
            return Err(DecodeError::BadMagic { offset: bytes.len(), expected: Self::END_HEADER, found: String::new() });
        };
        let mut at = header_end + end.len();
        let mut out = Self { events: Vec::new(), invalid: 0, skipped_packets: 0 };
        let mut previous: Option<u64> = None;
        while at < bytes.len() {
            let head = slice_at(bytes, at, Self::PACKET_HEADER)?;
            let word = |k: usize| u32::from_le_bytes([head[k], head[k + 1], head[k + 2], head[k + 3]]);
            let event_type = u16::from_le_bytes([head[0], head[1]]);
            let (size, ts_offset, overflow, capacity, number, valid) = (word(4), word(8), word(12), word(16), word(20), word(24));
            for (field, value) in [("eventSize", size), ("eventTSOverflow", overflow), ("eventNumber", number), ("eventValid", valid)] {
                if value > i32::MAX as u32 {
                    return Err(DecodeError::FieldOutOfRange { offset: at, field, value: u64::from(value), max: i32::MAX as u64 });
                }
            }
            if capacity != number {
                return Err(DecodeError::CountMismatch { offset: at + 16, declared: u64::from(capacity), actual: u64::from(number) });
            }
            let body_len = (size as usize).checked_mul(number as usize).ok_or(DecodeError::Truncated { offset: at, need: usize::MAX, have: 0 })?;
            let body = slice_at(bytes, at + Self::PACKET_HEADER, body_len)?;
            if event_type == Self::POLARITY {
                if size != 8 || ts_offset != 4 {
                    return Err(DecodeError::UnsupportedRecordLayout { offset: at, record_type: 1, record_size: size.min(255) as u8 });
                }
                let mut found_valid = 0u32;
                for (k, e) in body.as_chunks::<8>().0.iter().enumerate() {
                    let data = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    let stamp = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
                    let here = at + Self::PACKET_HEADER + 8 * k;
                    if stamp > i32::MAX as u32 {
                        return Err(DecodeError::FieldOutOfRange { offset: here + 4, field: "timestamp", value: u64::from(stamp), max: i32::MAX as u64 });
                    }
                    if data & 1 == 0 {
                        out.invalid += 1;
                        continue;
                    }
                    found_valid += 1;
                    let t = (u64::from(overflow) << 31) | u64::from(stamp);
                    if let Some(p) = previous.filter(|&p| t < p) {
                        return Err(DecodeError::NonMonotonicTimestamp { offset: here, previous: p, found: t });
                    }
                    previous = Some(t);
                    let polarity = if data & 2 != 0 { Polarity::On } else { Polarity::Off };
                    out.events.push(AerEvent { t, x: ((data >> 17) & 0x7FFF) as u16, y: ((data >> 2) & 0x7FFF) as u16, polarity });
                }
                if found_valid != valid {
                    return Err(DecodeError::CountMismatch { offset: at + 24, declared: u64::from(valid), actual: u64::from(found_valid) });
                }
            } else {
                out.skipped_packets += 1;
            }
            at += Self::PACKET_HEADER + body_len;
        }
        Ok(out)
    }

    /// Encode events as one file of polarity packets of at most `per_packet` events, source 1, with
    /// a minimal header. The result decodes bit-exactly.
    ///
    /// Returns `None` for `per_packet` of zero, a coordinate past 15 bits, or events out of time
    /// order — the format has nowhere to put the first two and forbids the third.
    #[must_use]
    pub fn encode(events: &[AerEvent], per_packet: usize) -> Option<Vec<u8>> {
        if per_packet == 0 || per_packet > i32::MAX as usize || events.windows(2).any(|p| p[1].t < p[0].t) {
            return None;
        }
        let mut out = Vec::from(Self::MAGIC.as_bytes());
        out.extend_from_slice(Self::END_HEADER.as_bytes());
        // A packet has ONE overflow counter, so it is also cut where the counter changes.
        let mut start = 0;
        while start < events.len() {
            let overflow = events[start].t >> 31;
            let mut stop = start;
            while stop < events.len() && stop - start < per_packet && events[stop].t >> 31 == overflow {
                stop += 1;
            }
            let count = (stop - start) as u32;
            out.extend_from_slice(&Self::POLARITY.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            for word in [8u32, 4, u32::try_from(overflow).ok().filter(|o| *o <= i32::MAX as u32)?, count, count, count] {
                out.extend_from_slice(&word.to_le_bytes());
            }
            for e in &events[start..stop] {
                if e.x > 0x7FFF || e.y > 0x7FFF {
                    return None;
                }
                let data = (u32::from(e.x) << 17) | (u32::from(e.y) << 2) | (u32::from(e.polarity == Polarity::On) << 1) | 1;
                out.extend_from_slice(&data.to_le_bytes());
                out.extend_from_slice(&((e.t & 0x7FFF_FFFF) as u32).to_le_bytes());
            }
            start = stop;
        }
        Some(out)
    }
}

/// One labelled interval of a DVS128 Gesture recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureLabel {
    /// Class, `1..=11` as the dataset numbers them (11 is "other gestures").
    pub class: u8,
    /// First microsecond of the gesture.
    pub start_us: u64,
    /// Last microsecond of the gesture.
    pub end_us: u64,
}

/// The sensor of the DVS128 Gesture dataset is 128 × 128.
pub const GESTURE_SENSOR: (u16, u16) = (128, 128);
/// The dataset has eleven classes.
pub const GESTURE_CLASSES: u8 = 11;

/// Parse a DVS128 Gesture `*_labels.csv`: a header line `class,startTime_usec,endTime_usec` and one
/// line per gesture.
///
/// # Errors
///
/// [`DecodeError::BadMagic`] if the first line is not that header; [`DecodeError::MalformedFlatBuffer`]
/// — reused for "a text record that does not parse" — naming the byte offset of a line that does
/// not have three unsigned integers; [`DecodeError::FieldOutOfRange`] for a class outside
/// `1..=11` or an interval that ends before it starts.
pub fn gesture_labels(csv: &str) -> Result<Vec<GestureLabel>, DecodeError> {
    const HEADER: &str = "class,startTime_usec,endTime_usec";
    let mut lines = csv.lines();
    let first = lines.next().unwrap_or("");
    if first.trim_end() != HEADER {
        return Err(DecodeError::BadMagic { offset: 0, expected: HEADER, found: first.to_string() });
    }
    let mut at = first.len() + 1;
    let mut out = Vec::new();
    for line in lines {
        let here = at;
        at += line.len() + 1;
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.trim().split(',').collect();
        let parsed: Option<Vec<u64>> = if parts.len() == 3 { parts.iter().map(|p| p.trim().parse().ok()).collect() } else { None };
        let Some(v) = parsed else {
            return Err(DecodeError::MalformedFlatBuffer { offset: here, what: "a label line is not three unsigned integers" });
        };
        if v[0] == 0 || v[0] > u64::from(GESTURE_CLASSES) {
            return Err(DecodeError::FieldOutOfRange { offset: here, field: "class", value: v[0], max: u64::from(GESTURE_CLASSES) });
        }
        if v[2] < v[1] {
            return Err(DecodeError::FieldOutOfRange { offset: here, field: "startTime_usec", value: v[1], max: v[2] });
        }
        out.push(GestureLabel { class: v[0] as u8, start_us: v[1], end_us: v[2] });
    }
    Ok(out)
}

/// The events of one labelled gesture: those with `start_us <= t <= end_us` — BOTH ends included,
/// which is this function's choice and not something the dataset specifies. The events must be in
/// time order, which every decoder in this module guarantees; the slice is found by bisection.
/// Labels in the dataset are known to overlap in places (one recording's class 9 ends after its
/// class 10 begins), so two gestures' slices may share events; nothing here forbids that.
#[must_use]
pub fn gesture_events<'a>(events: &'a [AerEvent], label: &GestureLabel) -> &'a [AerEvent] {
    let from = events.partition_point(|e| e.t < label.start_us);
    let to = events.partition_point(|e| e.t <= label.end_us);
    &events[from..to.max(from)]
}

#[cfg(test)]
mod tests {
    use super::{
        Aedat2, Aedat2Layout, Aedat3, Aedat4, Aedat4Compression, Aedat4Packet, AerEvent, Dat, DecodeError,
        EncodeError, Evt2, Evt3, Flat, GestureLabel, Marker, MarkerKind, TrainMap, gesture_events, gesture_labels,
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

    /// The header/record seam, swept over **every row of both presets**.
    ///
    /// `AEDAT` 2.0 has no header terminator: the records start at the first byte that is not a
    /// header line, so a record whose first byte is `#` can be eaten as a comment. It is not
    /// hypothetical — the `DAVIS346` row field sits at bit 22, so row fields 140 to 143 put `0x23`
    /// in the big-endian address MSB; since the preset counts rows from the top, those are decoded
    /// rows 116 to 119. Measured before the fix: 4 of that sensor's 260 rows destroyed their own
    /// round trip, and a three-event stream beginning on row field 140 came back `Ok` with two
    /// events and two header lines. One seed and one first event is exactly what the bit-exactness
    /// test above samples, which is why this sweeps the coordinate that decides it.
    #[test]
    fn aedat2_round_trips_every_first_event_row_of_both_presets() {
        let header = vec![
            "!AER-DAT2.0".to_string(),
            " a comment line, as jAER writes several".to_string(),
            " with a non-ASCII byte: \u{b5}s".to_string(),
        ];
        for layout in [Aedat2Layout::DVS128, Aedat2Layout::DAVIS346] {
            let mut destroyed = Vec::new();
            for row in 0..layout.height {
                for x in [0u16, 1, layout.width - 1] {
                    let events = vec![
                        AerEvent { t: 10, x, y: row, polarity: Polarity::On },
                        AerEvent { t: 20, x: 0, y: 0, polarity: Polarity::Off },
                        AerEvent { t: 0x0A0A_0A0A, x: 1, y: 1, polarity: Polarity::On },
                    ];
                    let bytes = Aedat2::encode(&events, layout, &header).expect("encodable");
                    match Aedat2::decode(&bytes, layout) {
                        Ok(back) if back.events == events && back.header == header => {}
                        other => destroyed.push((row, x, format!("{other:?}"))),
                    }
                }
            }
            assert!(
                destroyed.is_empty(),
                "{}: {} of {} first-event rows lost their round trip, first {:?}",
                layout.source,
                destroyed.len(),
                layout.height,
                destroyed.first()
            );
        }
    }

    /// The seam from the other side: a record that *is* a `#` must not be read as a comment, and a
    /// comment that is text must still be read as one.
    #[test]
    fn a_binary_record_beginning_with_the_prefix_byte_is_not_a_header_line() {
        // Row 119 of a DAVIS346 is row field 259 - 119 = 140, and 140 << 22 = 0x23000000 puts the
        // character '#' in the address MSB.
        let hazard = AerEvent { t: 10, x: 5, y: 119, polarity: Polarity::On };
        let bytes = Aedat2::encode(&[hazard], Aedat2Layout::DAVIS346, &[]).expect("encodable");
        assert_eq!(bytes[14], b'#', "the fixture must actually contain the hazard");
        let back = Aedat2::decode(&bytes, Aedat2Layout::DAVIS346).expect("decodable");
        assert_eq!(back.events, vec![hazard]);
        assert_eq!(back.header, vec!["!AER-DAT2.0".to_string()], "one line, not two");

        // And the rule that separates them is "is it text", so a text comment still parses --
        // including one that is not ASCII, and one that is empty.
        let lines = vec![
            "!AER-DAT2.0".to_string(),
            String::new(),
            "\ttabbed".to_string(),
            " 8 \u{b5}s per tick".to_string(),
        ];
        let b = Aedat2::encode(&[hazard], Aedat2Layout::DAVIS346, &lines).expect("encodable");
        let d = Aedat2::decode(&b, Aedat2Layout::DAVIS346).expect("decodable");
        assert_eq!(d.header, lines);
        assert_eq!(d.events, vec![hazard]);
    }

    /// The two things a 32-bit wrapping clock cannot be unwrapped back to, refused by both
    /// formats that carry one.
    ///
    /// Measured before this was fixed: two events exactly 2^32 us apart came back **0 us** apart,
    /// in order, with no error, and a single event at 2^32 + 12,345 came back at 12,345 — from
    /// both `AEDAT` 2.0 and `.dat`, while `Flat`, with 64 bits of time, returned it intact.
    #[test]
    fn aedat2_refuses_a_time_its_own_decoder_could_not_recover() {
        let at = |t: u64| AerEvent { t, x: 1, y: 2, polarity: Polarity::On };
        for (name, enc) in [
            ("aedat2", &(|e: &[AerEvent]| Aedat2::encode(e, Aedat2Layout::DVS128, &[]))
                as &dyn Fn(&[AerEvent]) -> Result<Vec<u8>, EncodeError>),
            ("dat", &(|e: &[AerEvent]| Dat::encode(e, &[], 0x00))),
        ] {
            // A first timestamp past the counter: the accumulator starts at zero and the high bits
            // are nowhere on the wire.
            assert!(
                matches!(
                    enc(&[at(1u64 << 32)]),
                    Err(EncodeError::FieldOutOfRange {
                        index: 0,
                        field: "first timestamp",
                        value: 4_294_967_296,
                        max: 4_294_967_295
                    })
                ),
                "{name} wrote a first timestamp it could not read back"
            );
            // Exactly one counter period apart reads as no gap at all.
            for gap in [1u64 << 32, (1u64 << 32) + 10, 1u64 << 31] {
                assert!(
                    matches!(
                        enc(&[at(0), at(gap)]),
                        Err(EncodeError::GapTooLarge { index: 1, max: 2_147_483_647, .. })
                    ),
                    "{name} wrote a gap of {gap} us"
                );
            }
            // The largest first timestamp, and the largest gap, at a phase that straddles the
            // wrap: both encode AND come back exactly.
            let last = (1u64 << 32) - 1;
            let ok = [at(last - 99), at(last - 99 + Aedat2::MAX_GAP_US)];
            let bytes = enc(&ok).expect("the stated bounds encode");
            let back = if name == "aedat2" {
                Aedat2::decode(&bytes, Aedat2Layout::DVS128).expect("decodable").events
            } else {
                Dat::decode(&bytes).expect("decodable").events
            };
            assert_eq!(back, ok, "{name} at the bound, across the wrap");
            assert_eq!(back[1].t - back[0].t, Aedat2::MAX_GAP_US);
        }
        assert_eq!(Aedat2::MAX_GAP_US, Dat::MAX_GAP_US);
        assert_eq!(Aedat2::MAX_TIME_US, Dat::MAX_TIME_US);
    }

    /// And the gap just past the bound is one the decoder genuinely refuses — so the encoder can
    /// no longer write a file its own decoder rejects.
    #[test]
    fn a_gap_of_half_the_counter_is_refused_by_the_decoder_too() {
        let p = 0xFFFF_FF9Cu32;
        let q = p.wrapping_add(1 << 31);
        let bytes = aedat2_bytes(&[(0, p), (0, q)]);
        assert!(
            matches!(
                Aedat2::decode(&bytes, Aedat2Layout::DVS128),
                Err(DecodeError::NonMonotonicTimestamp { .. })
            ),
            "a backwards jump of exactly half the range is ambiguous and must not be a wrap"
        );
        // One microsecond less, and it is a wrap the decoder does recover.
        let bytes = aedat2_bytes(&[(0, p), (0, q.wrapping_sub(1))]);
        let back = Aedat2::decode(&bytes, Aedat2Layout::DVS128).expect("decodable");
        assert_eq!(back.events[1].t - back.events[0].t, Aedat2::MAX_GAP_US);
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
        assert_eq!(back.compression, Aedat4Compression::None);
        assert_eq!(back.events().expect("all uncompressed"), events);
        assert!(back.packets.iter().all(|p| p.stream_id == 7));
        // And the container itself is byte-stable, which is what lets a pipeline re-save a file
        // it only partly understands.
        assert_eq!(back.encode().expect("serialisable"), bytes);
    }

    /// Every format decodes to the same events from the same input. This is the check that the six
    /// codecs share one semantics rather than six nearly-identical ones.
    #[test]
    fn all_six_formats_agree_on_the_same_events() {
        // Generated at the DAVIS346 geometry so that all six formats can genuinely hold it. At
        // 640x480 the AEDAT 2.0 arm could only ever assert its own refusal, which made the
        // "all six" claim a five-format check.
        let (w, h) = (Aedat2Layout::DAVIS346.width, Aedat2Layout::DAVIS346.height);
        let events = synth(2_000, 0x17, w, h, 250);
        let header = vec![" Width 346".to_string(), " Height 260".to_string()];
        let via_flat = Flat::decode(&Flat::encode(&events, w, h).unwrap()).unwrap().events;
        let via_evt2 = Evt2::decode(&Evt2::encode(&events).unwrap()).unwrap().events;
        let via_evt3 = Evt3::decode(&Evt3::encode(&events, true).unwrap()).unwrap().events;
        let via_dat = Dat::decode(&Dat::encode(&events, &header, 0x00).unwrap()).unwrap().events;
        let via_a2 = Aedat2::decode(
            &Aedat2::encode(&events, Aedat2Layout::DAVIS346, &[]).unwrap(),
            Aedat2Layout::DAVIS346,
        )
        .unwrap()
        .events;
        let via_a4 = Aedat4::raw_from_events(&events, 1024, 0).unwrap().events().unwrap();
        for (name, got) in [
            ("flat", &via_flat),
            ("evt2", &via_evt2),
            ("evt3", &via_evt3),
            ("dat", &via_dat),
            ("aedat2", &via_a2),
            ("aedat4", &via_a4),
        ] {
            assert_eq!(got, &events, "{name} disagrees");
        }
        // And the refusal that the old version of this test asserted instead is still asserted,
        // where it belongs: a 640x480 stream does not fit the DAVIS346 layout, and says so.
        let big = synth(4, 0x17, 640, 480, 10);
        assert!(matches!(
            Aedat2::encode(&big, Aedat2Layout::DAVIS346, &[]),
            Err(EncodeError::FieldOutOfRange { .. })
        ));
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

    /// `AEDAT` 2.0, `DVS128`: row in bits 8-14, column in bits 1-7, both counted from the far edge,
    /// polarity in bit 0 with a clear bit meaning `On`, whole record big-endian.
    ///
    /// The mapping is `dv-processing`'s `DVS128` parser at its defaults — `x = (WIDTH - 1) - x`,
    /// `y = (HEIGHT - 1) - y`, and `On` when the polarity bit is 0 — applied by hand. Column 10
    /// becomes 127 - 10 = 117 at bits 1-7, i.e. 234; row 20 becomes 127 - 20 = 107 at bits 8-14,
    /// i.e. 27,392; so the address is 27,626 = `0x6BEA`, and the record is `00 00 6B EA` then the
    /// timestamp, both big-endian — the only big-endian format in this module. Through 0.22.0 the
    /// row was not turned round and this record was `00 00 14 EA`.
    #[test]
    fn aedat2_records_match_the_jaer_field_table() {
        let e = AerEvent { t: 0x1122_3344, x: 10, y: 20, polarity: Polarity::On };
        let bytes = Aedat2::encode(&[e], Aedat2Layout::DVS128, &[]).expect("encodable");
        assert_eq!(&bytes[bytes.len() - 8..], &[0x00, 0x00, 0x6B, 0xEA, 0x11, 0x22, 0x33, 0x44]);
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
    /// The fixture used to cross twice while the doc above it said three, which is the sort of
    /// number a reader checks the code against; it now crosses three times and the last event is
    /// 3 x 2^24 + 1.
    #[test]
    fn evt3_reconstructs_timestamps_across_repeated_rollovers() {
        let s = evt3_words(&[
            0x8000, 0x6001, 0x0005, 0x2000 | 10, // t = 1
            0x8FFF, 0x6FFF, 0x2000 | 11,         // t = 16,773,120 + 4,095 = 16,777,215
            0x8000, 0x6002, 0x2000 | 12,         // wrap 1: t = 16,777,216 + 2
            0x8FFF, 0x6FF0, 0x2000 | 13,         // t = 16,777,216 + 16,773,120 + 4,080
            0x8000, 0x6000, 0x2000 | 14,         // wrap 2: t = 33,554,432
            0x8FFF, 0x6FF0, 0x2000 | 15,         // t = 33,554,432 + 16,773,120 + 4,080
            0x8000, 0x6001, 0x2000 | 16,         // wrap 3: t = 50,331,648 + 1
        ]);
        let got = Evt3::decode(&s).expect("decodable");
        let times: Vec<u64> = got.events.iter().map(|e| e.t).collect();
        assert_eq!(
            times,
            vec![1, 16_777_215, 16_777_218, 33_554_416, 33_554_432, 50_331_632, 50_331_649]
        );
        assert_eq!(times[6], 3 * Evt3::WRAP_US + 1, "three wraps, counted");
        assert!(monotonic(&got.events), "a missed wrap shows up here as a sawtooth");
        // And the columns rode along correctly, so this is not a test of time alone.
        assert_eq!(
            got.events.iter().map(|e| e.x).collect::<Vec<_>>(),
            vec![10, 11, 12, 13, 14, 15, 16]
        );
    }

    /// [`Evt3::MAX_FIRST_TIME_US`] is an inclusive bound: that value encodes, the next does not.
    ///
    /// The doc said "at or past" while `fit` refuses only `>`, so the constant named one thing and
    /// the code did another by one microsecond. The code is right — 2^24 - 1 is all 24 bits of
    /// time set and encodes exactly — so this pins it from both sides.
    #[test]
    fn evt3_encodes_its_largest_first_timestamp_and_refuses_the_next() {
        let at = |t: u64| [AerEvent { t, x: 3, y: 4, polarity: Polarity::Off }];
        let last = at(Evt3::MAX_FIRST_TIME_US);
        let bytes = Evt3::encode(&last, false).expect("2^24 - 1 is encodable");
        assert_eq!(Evt3::decode(&bytes).unwrap().events, last);
        assert!(matches!(
            Evt3::encode(&at(Evt3::MAX_FIRST_TIME_US + 1), false),
            Err(EncodeError::FieldOutOfRange {
                index: 0,
                field: "first timestamp",
                value: 16_777_216,
                max: 16_777_215
            })
        ));
        assert_eq!(Evt3::MAX_FIRST_TIME_US + 1, Evt3::WRAP_US);
    }

    /// An `EVT` 3.0 event word carries no time of its own, so an event before any time word is
    /// dated zero — the stream did not say, and this decoder does not guess.
    ///
    /// Documented on [`Marker::t`] and pinned here, because "t = 0" reads exactly like the start
    /// of a recording and a caller who resumed from an arbitrary offset needs to know it is not.
    #[test]
    fn evt3_dates_an_event_before_any_time_word_at_zero() {
        let s = evt3_words(&[0x0005, 0x2000 | 9, 0x8001, 0x6002, 0x2000 | 11]);
        let got = Evt3::decode(&s).expect("decodable");
        assert_eq!(
            got.events,
            vec![
                AerEvent { t: 0, x: 9, y: 5, polarity: Polarity::Off },
                AerEvent { t: 4_098, x: 11, y: 5, polarity: Polarity::Off },
            ]
        );
        assert!(got.markers.is_empty(), "and nothing is reported that was not on the wire");
    }

    /// `CONTINUED` words are reported as continuations rather than dropped or mistaken for events.
    ///
    /// Both `EVT` 3.0 continuation opcodes and the `EVT` 2.0 one, which are the tail of a
    /// multi-word payload this implementation does not interpret. [`MarkerKind::Continued`] has no
    /// other test.
    #[test]
    fn continued_words_are_reported_as_continuations() {
        let s = evt3_words(&[
            (Evt3::TIME_HIGH << 12),
            (Evt3::TIME_LOW << 12) | 1,
            (Evt3::CONTINUED_4 << 12) | 0xA,
            (Evt3::CONTINUED_12 << 12) | 0xBCD,
        ]);
        let f = Evt3::decode(&s).expect("decodable");
        assert!(f.events.is_empty());
        assert_eq!(
            f.markers,
            vec![
                Marker { offset: 4, kind: MarkerKind::Continued, raw: 0x700A, t: 1 },
                Marker { offset: 6, kind: MarkerKind::Continued, raw: 0xFBCD, t: 1 },
            ]
        );
        let mut e2 = Evt2::encode(&[AerEvent { t: 100, x: 1, y: 1, polarity: Polarity::On }])
            .unwrap();
        e2.extend_from_slice(&((u32::from(Evt2::CONTINUED) << 28) | 0x1234).to_le_bytes());
        e2.extend_from_slice(&((u32::from(Evt2::OTHERS) << 28) | 0x5678).to_le_bytes());
        let f2 = Evt2::decode(&e2).expect("decodable");
        assert_eq!(
            f2.markers.iter().map(|m| m.kind).collect::<Vec<_>>(),
            vec![MarkerKind::Continued, MarkerKind::Other]
        );
        assert_eq!(f2.markers[0].raw, (u64::from(Evt2::CONTINUED) << 28) | 0x1234);
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

    /// What a sweep of malformed input actually exercised, **per decoder**.
    ///
    /// Counted and asserted on, because a totality test is the easiest kind to make vacuous: a
    /// decoder that returned `Err` for everything would pass "it never panics" perfectly, and so
    /// would a test whose fixtures were all too short to reach the state machine.
    ///
    /// Per decoder rather than summed, because a single global total cannot see one decoder
    /// regressing. Measured on the truncation sweep: of 1,341 successes, `Flat` contributed 3 and
    /// `AEDAT` 4.0 contributed 17, so either could stop succeeding altogether and a floor of 500
    /// on the sum would still pass with 976.
    #[derive(Default, Debug, Clone, Copy)]
    struct Tally {
        ok: u64,
        err: u64,
        events: u64,
    }

    /// The decoder slots, in the order [`decode_every_way`] runs them.
    const DECODERS: [&str; 7] = [
        "aedat2-dvs128",
        "aedat2-davis346",
        "evt2",
        "evt3",
        "dat",
        "flat",
        "aedat4",
    ];

    /// Which slot of [`DECODERS`] is the decoder a corpus fixture was written by.
    fn own_decoder(fixture: &str) -> usize {
        match fixture {
            "aedat2" => 0,
            "evt2" => 2,
            "evt3-plain" | "evt3-vect" => 3,
            "dat" => 4,
            "flat" => 5,
            "aedat4" => 6,
            other => panic!("fixture {other} has no decoder slot"),
        }
    }

    #[derive(Default, Debug)]
    struct Sweep {
        per: [Tally; DECODERS.len()],
    }

    impl Sweep {
        fn total(&self) -> Tally {
            let mut t = Tally::default();
            for d in &self.per {
                t.ok += d.ok;
                t.err += d.err;
                t.events += d.events;
            }
            t
        }

        fn add(&mut self, other: &Self) {
            for (a, b) in self.per.iter_mut().zip(other.per.iter()) {
                a.ok += b.ok;
                a.err += b.err;
                a.events += b.events;
            }
        }
    }

    /// Decode with every decoder and assert the two properties that must hold for ANY input: the
    /// call returns, and if it returns events they are monotonically non-decreasing in time.
    ///
    /// Deliberately runs every decoder over every buffer, not just the matching one: reading a
    /// file with the wrong decoder is a routine accident and must fail rather than crash.
    fn decode_every_way(bytes: &[u8], sweep: &mut Sweep) {
        let mut slot = 0usize;
        let mut note = |r: Result<Vec<AerEvent>, DecodeError>, sweep: &mut Sweep| {
            let t = &mut sweep.per[slot];
            match r {
                Ok(e) => {
                    assert!(monotonic(&e), "{} returned a non-monotonic list", DECODERS[slot]);
                    t.ok += 1;
                    t.events += e.len() as u64;
                }
                Err(_) => t.err += 1,
            }
            slot += 1;
        };
        for layout in [Aedat2Layout::DVS128, Aedat2Layout::DAVIS346] {
            note(Aedat2::decode(bytes, layout).map(|f| f.events), sweep);
        }
        note(Evt2::decode(bytes).map(|f| f.events), sweep);
        note(Evt3::decode(bytes).map(|f| f.events), sweep);
        note(Dat::decode(bytes).map(|f| f.events), sweep);
        note(Flat::decode(bytes).map(|f| f.events), sweep);
        note(Aedat4::decode(bytes).and_then(|f| f.events()), sweep);
        assert_eq!(slot, DECODERS.len(), "every decoder slot is named in DECODERS");
    }

    /// The test that makes this module safe to point at a file.
    ///
    /// Every prefix of every valid stream, for every decoder. A panic anywhere fails the test,
    /// because a panicking decoder is how a truncated recording — the normal result of a power
    /// loss or a cancelled copy — takes down whatever is reading it.
    #[test]
    fn every_truncation_of_every_format_errors_instead_of_panicking() {
        let mut all = Sweep::default();
        for (name, bytes) in corpus() {
            assert!(bytes.len() > 100, "{name} is too short to be a useful fixture");
            let mut sweep = Sweep::default();
            for cut in 0..=bytes.len() {
                decode_every_way(&bytes[..cut], &mut sweep);
            }
            // PER FIXTURE AND PER DECODER, not on the sum. The decoder that wrote the fixture must
            // both accept it — at minimum the untruncated file — and refuse some prefix of it, or
            // it has stopped reading its own format and no global floor would say so.
            let own = sweep.per[own_decoder(name)];
            assert!(own.ok >= 1, "{name}: its own decoder accepted nothing: {own:?}");
            assert!(own.err >= 1, "{name}: its own decoder refused nothing: {own:?}");
            assert!(own.events >= 1, "{name}: its own decoder produced no events: {own:?}");
            if let Some(i) = PRINT_SWEEP.then_some(name) {
                println!("{i}: {:?}", sweep.per);
            }
            all.add(&sweep);
        }
        // And every decoder, over the whole sweep, has gone both ways. Measured on this fixed
        // corpus, ok / err / events per slot:
        //   aedat2-dvs128    121 / 8,217 / 7,260     aedat2-davis346  121 / 8,217 / 7,260
        //   evt2             227 / 8,111 / 13,108    evt3             735 / 7,603 / 43,314
        //   dat              121 / 8,217 / 7,260     flat               1 / 8,337 /    120
        //   aedat4             5 / 8,333 /   312
        // Flat's single success is not a weakness: its header declares the record count, so the
        // only prefix of a Flat file that decodes is the whole file. That is exactly why the floor
        // below is 1 and why the per-fixture check above — own decoder, own fixture — is the one
        // carrying the weight.
        for (i, name) in DECODERS.iter().enumerate() {
            let t = all.per[i];
            assert!(t.ok >= 1, "{name} accepted nothing in the whole sweep: {t:?}");
            assert!(t.err > 1_000, "{name} refused only {} of the sweep: {t:?}", t.err);
            assert!(t.events >= 100, "{name} produced only {} events: {t:?}", t.events);
        }
        let total = all.total();
        // Totals, measured on this fixed corpus: 1,331 successes, 57,035 refusals, 78,634 events.
        assert!(total.err > 5_000, "only {} refusals; is anything being refused?", total.err);
        assert!(total.ok > 500, "only {} successes; are the fixtures reaching the decoders?", total.ok);
        assert!(total.events > 10_000, "only {} events decoded in the whole sweep", total.events);
    }

    /// Flip to print the per-decoder matrices the two sweeps' comments quote, then flip back.
    const PRINT_SWEEP: bool = false;

    /// Random bytes of random lengths, seeded so the failure is reproducible.
    ///
    /// The point of the counts here is the second half of the claim: noise is *almost* always
    /// refused, and the "almost" is measured rather than asserted away.
    #[test]
    fn random_bytes_never_panic() {
        let mut r = Rng::new(0x9E37_79B9);
        let mut sweep = Sweep::default();
        let mut empty_sweep = Sweep::default();
        let mut empties = 0u64;
        for _ in 0..4_000 {
            let n = r.below(600) as usize;
            let mut buf = Vec::with_capacity(n);
            for _ in 0..n {
                buf.push((r.next_u32() & 0xFF) as u8);
            }
            if buf.is_empty() {
                empties += 1;
                decode_every_way(&buf, &mut empty_sweep);
            } else {
                decode_every_way(&buf, &mut sweep);
            }
        }
        // The zero-length draws are separated because they are not noise: an empty buffer is an
        // empty stream, and which decoders accept one is a property of the formats rather than of
        // this seed. Exactly two do — EVT 2.0 and EVT 3.0, whose streams are a bare word array
        // with no header to be missing. The other five require a magic, a descriptor or a fixed
        // header and are right to refuse. Measured on this seed: 11 zero-length draws.
        assert_eq!(empties, 11, "the seed draws 11 empty buffers");
        for (i, name) in DECODERS.iter().enumerate() {
            let t = empty_sweep.per[i];
            let accepts_empty = *name == "evt2" || *name == "evt3";
            assert_eq!(
                t.ok,
                if accepts_empty { empties } else { 0 },
                "{name} on an empty buffer: {t:?}"
            );
            assert_eq!(t.events, 0, "{name} invented events from an empty buffer");
        }
        // And the real claim, on the 3,989 non-empty noise buffers. Noise IS occasionally accepted,
        // and the previous version of this comment said the opposite — that the only successes
        // were the empty buffers, "which every decoder is right to accept". Both halves were
        // wrong. Measured on this seed, across 27,923 decodes of non-empty noise:
        //
        //   EVT 2.0 accepted 4 and produced 2 events, both from one 28-byte buffer;
        //   EVT 3.0 accepted 6 and produced none — they are buffers of time and marker words;
        //   the other five decoders accepted nothing at all, 3,989 refusals each.
        //
        // Ten successes per 4,000 draws is this module's weakest refusal, and it is what the two
        // formats with no magic, no header and no length can do. Pinned exactly, so that a decoder
        // which started accepting noise in bulk fails here rather than passing a `> 10_000`.
        let noise = sweep.total();
        if PRINT_SWEEP {
            println!("NOISE per decoder: {:?}", sweep.per);
        }
        assert_eq!(noise.ok, 10, "noise accepted {} times: {sweep:?}", noise.ok);
        assert_eq!(noise.events, 2, "noise decoded to {} events: {sweep:?}", noise.events);
        assert_eq!(sweep.per[2].ok, 4, "EVT 2.0 on noise: {:?}", sweep.per[2]);
        assert_eq!(sweep.per[3].ok, 6, "EVT 3.0 on noise: {:?}", sweep.per[3]);
        assert!(noise.err > 10_000, "{sweep:?}");
        for (i, name) in DECODERS.iter().enumerate() {
            let t = sweep.per[i];
            assert!(t.err > 3_000, "{name} refused only {} noise buffers: {t:?}", t.err);
            if *name != "evt2" && *name != "evt3" {
                assert_eq!(t.ok, 0, "{name} accepted a random buffer: {t:?}");
            }
        }
    }

    /// Valid streams with bytes flipped. This reaches deeper into each decoder than random noise
    /// does — the header still parses, so the damage lands in the state machine.
    #[test]
    fn mutated_streams_never_panic() {
        let mut r = Rng::new(0xC0FF_EE01);
        let mut sweep = Sweep::default();
        for (_, bytes) in corpus() {
            for _ in 0..500 {
                let mut m = bytes.clone();
                for _ in 0..1 + r.below(6) {
                    let at = r.below(m.len() as u32) as usize;
                    m[at] ^= (r.next_u32() & 0xFF) as u8;
                }
                decode_every_way(&m, &mut sweep);
            }
        }
        // Both outcomes, in bulk: damaged files that still decode (the mutation landed in a
        // coordinate) and damaged files that are refused (it landed in an opcode or a length).
        // Measured on this seed: 1,124 successes, 23,376 refusals, 129,633 events decoded.
        let tally = sweep.total();
        assert!(tally.ok > 200, "{sweep:?}");
        assert!(tally.err > 1_000, "{sweep:?}");
        assert!(tally.events > 50_000, "{sweep:?}");
        // Per decoder, so that one of the seven going dark cannot hide behind the other six.
        for (i, name) in DECODERS.iter().enumerate() {
            assert!(sweep.per[i].err > 100, "{name} refused almost nothing: {:?}", sweep.per[i]);
        }
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
    ///
    /// The `EVT` 3.0 stream below fails at byte 0 because it does not begin with `#` at all, which
    /// is the easy case and the only one this test used to cover. The cases that matter are the
    /// two files that **do**: `AEDAT` 3.1 and `AEDAT` 4.0 are the same vendor, the same `.aedat`
    /// extension and completely different record layouts, and a magic of `"!AER-DAT"` with no
    /// version digit admits both. Measured before the digit was added: a jAER `AEDAT` 3.1 file
    /// decoded as 200 well-formed `AEDAT` 2.0 events and an `AEDAT` 4.0 file as 100, both `Ok`.
    /// `DVS128` makes this the only defence it has — every 32-bit pattern is a valid address for a
    /// 15-bit layout, so no record of one can be refused.
    #[test]
    fn a_missing_magic_stops_one_format_being_read_as_another() {
        for magic in ["#!AER-DAT3.1\r\n", "#!AER-DAT4.0\r\n", "#!AER-DAT1.0\r\n"] {
            let mut f = magic.as_bytes().to_vec();
            for i in 0..200u32 {
                f.extend_from_slice(&(i * 7).to_be_bytes());
                f.extend_from_slice(&(i * 100).to_be_bytes());
            }
            for layout in [Aedat2Layout::DVS128, Aedat2Layout::DAVIS346] {
                match Aedat2::decode(&f, layout) {
                    Err(DecodeError::BadMagic { offset: 0, expected, .. }) => {
                        assert_eq!(expected, "#!AER-DAT2.0");
                    }
                    other => panic!(
                        "{magic:?} was read as AEDAT 2.0 by {}: {:?}",
                        layout.source,
                        other.map(|d| d.events.len())
                    ),
                }
            }
        }
        // And the version this decoder does read is still read, whatever follows the digit.
        for magic in ["!AER-DAT2.0", "!AER-DAT2"] {
            let e = [AerEvent { t: 1, x: 2, y: 3, polarity: Polarity::On }];
            let bytes =
                Aedat2::encode(&e, Aedat2Layout::DVS128, &[magic.to_string()]).expect("encodable");
            let back = Aedat2::decode(&bytes, Aedat2Layout::DVS128).expect("decodable");
            assert_eq!(back.header[0], magic);
            assert_eq!(back.events, e);
        }

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

    /// A `DVS128` address with a bit set above its 15-bit field is a marker, not a pixel.
    ///
    /// This preset used to carry `special_mask: 0`, which disables the check entirely. Measured
    /// then: addresses `1 << 15`, `1 << 31` and `0xFFFFFFFF` decoded as three ordinary pixel
    /// events at columns 127, 127 and 0, with zero markers — the exact outcome the [`Marker`] doc
    /// calls destroying the synchronisation evidence while leaving the event count unchanged.
    #[test]
    fn aedat2_dvs128_reports_an_address_outside_its_15_bits_as_a_marker() {
        let bytes = aedat2_bytes(&[
            (1 << 15, 100),
            (1 << 31, 200),
            (0xFFFF_FFFF, 300),
            (((127 - 20) << 8) | ((127 - 10) << 1) | 1, 400),
        ]);
        let f = Aedat2::decode(&bytes, Aedat2Layout::DVS128).expect("decodable");
        assert_eq!(f.markers.len(), 3, "{:?}", f.events);
        assert_eq!(
            f.markers,
            vec![
                Marker { offset: 14, kind: MarkerKind::Other, raw: 1 << 15, t: 100 },
                Marker { offset: 22, kind: MarkerKind::Other, raw: 1 << 31, t: 200 },
                Marker { offset: 30, kind: MarkerKind::Other, raw: 0xFFFF_FFFF, t: 300 },
            ]
        );
        // The ordinary record beside them still decodes, so this is not a blanket refusal: column
        // field 117 and row field 107, counted from the right and bottom edges, are column 10 and
        // row 20; polarity bit set.
        assert_eq!(
            f.events,
            vec![AerEvent { t: 400, x: 10, y: 20, polarity: Polarity::Off }]
        );
        assert_eq!(Aedat2Layout::DVS128.special_mask, 0xFFFF_8000);
        assert_eq!(Aedat2Layout::DVS128.overlap(), 0, "and the mask does not collide with a field");
    }

    /// A `DAVIS` `DVS` word with bit 10 set is an external-input event: a trigger, not a pixel.
    ///
    /// `jAER`'s `DavisChip.java` has `EVENT_TYPE_SHIFT = 10` and `EXTERNAL_INPUT_EVENT_ADDR = 1 <<
    /// EVENT_TYPE_SHIFT`, with falling, rising and pulse at `+ 2`, `+ 3` and `+ 4`; iniVation's
    /// `AEDAT` 2.0 page gives sub-types 01 and 11 in bits 11-10 as "External Event". Through 0.22.0
    /// the preset's special mask was bit 31 alone and every one of these words decoded as a pixel.
    #[test]
    fn a_davis346_external_input_word_is_a_trigger_not_a_pixel() {
        let ext = 1u32 << 10;
        let bytes = aedat2_bytes(&[
            (ext | 2, 100),             // falling edge, sub-type 01
            (ext | 3, 200),             // rising edge
            (ext | (1 << 11) | 4, 300), // pulse, sub-type 11, "Same as 01"
            ((1 << 31) | ext, 400),     // an APS or IMU word, whose bits 11-10 mean something else
            (0, 500),                   // an ordinary pixel beside them
        ]);
        let f = Aedat2::decode(&bytes, Aedat2Layout::DAVIS346).expect("decodable");
        let trigger = MarkerKind::ExternalTrigger;
        assert_eq!(
            f.markers,
            vec![
                Marker { offset: 14, kind: trigger, raw: u64::from(ext | 2), t: 100 },
                Marker { offset: 22, kind: trigger, raw: u64::from(ext | 3), t: 200 },
                Marker { offset: 30, kind: trigger, raw: u64::from(ext | (1 << 11) | 4), t: 300 },
                Marker { offset: 38, kind: MarkerKind::Other, raw: u64::from((1u32 << 31) | ext), t: 400 },
            ]
        );
        assert_eq!(f.markers[1].raw & 7, 3, "the edge rides in the low three bits");
        // Address zero is row field 0 and column field 0: the last row and the last column.
        assert_eq!(f.events, vec![AerEvent { t: 500, x: 345, y: 259, polarity: Polarity::Off }]);
        assert_eq!(Aedat2Layout::DAVIS346.special_mask, 0x8000_0400);
        assert_eq!(Aedat2Layout::DAVIS346.trigger_mask, 0x0000_0400);
        assert_eq!(Aedat2Layout::DVS128.trigger_mask, 0, "the DVS128 preset calls no marker a trigger");
    }

    /// The `DAVIS346` column is `jAER`'s, and its row counts from the top.
    ///
    /// `jAER`'s `DavisEventExtractor` decodes `e.x = (short) (sx1 - ((data & DavisChip.XMASK) >>>
    /// DavisChip.XSHIFT))` with `sx1 = getChip().getSizeX() - 1`, 345, and leaves the row field
    /// counting from the bottom, where iniVation's `AEDAT` 2.0 page puts "(0, 0)". Every expected
    /// coordinate below is that formula applied by hand to a raw field, then the row turned round:
    /// `x = 345 - raw_x`, `y = 259 - raw_y`. Through 0.22.0 the preset returned `raw_x` and `raw_y`
    /// unchanged — the mirror image of `jAER` in `x`, upside down in `y` — and setting its
    /// `x_invert` would not have helped, because the mirror was then about 1023.
    #[test]
    fn the_davis346_preset_mirrors_the_column_as_jaer_does_and_counts_rows_from_the_top() {
        let layout = Aedat2Layout::DAVIS346;
        for (raw_x, raw_y) in [(0u32, 0u32), (345, 259), (215, 164), (1, 258)] {
            let addr = (raw_x << 12) | (raw_y << 22);
            let f = Aedat2::decode(&aedat2_bytes(&[(addr, 5)]), layout).expect("inside the array");
            let want = (u16::try_from(345 - raw_x).unwrap(), u16::try_from(259 - raw_y).unwrap());
            assert_eq!((f.events[0].x, f.events[0].y), want, "fields ({raw_x}, {raw_y})");
            // And the encoder puts the same fields back. Bit 11 is clear, which this preset reads
            // as Off and writes for Off.
            let back = Aedat2::encode(&f.events, layout, &[]).expect("encodable");
            assert_eq!(&back[back.len() - 8..back.len() - 4], &addr.to_be_bytes());
        }
        // A field past the array is refused, and reported as the field on the wire: jAER would
        // decode column field 346 as column -1, which has no coordinate to report.
        for (addr, field, value, max) in [(346u32 << 12, "column", 346u64, 345u64), (260 << 22, "row", 260, 259)] {
            match Aedat2::decode(&aedat2_bytes(&[(addr, 5)]), layout) {
                Err(DecodeError::FieldOutOfRange { offset, field: f, value: v, max: m }) => {
                    assert_eq!((offset, f, v, m), (14, field, value, max));
                }
                other => panic!("{field} field {value} was decoded: {other:?}"),
            }
        }
        // With no geometry stated there is no edge to mirror about, and the mirror is about the
        // top of the field: the one case in which the old rule still applies.
        let free = Aedat2Layout { width: 0, height: 0, ..layout };
        let f = Aedat2::decode(&aedat2_bytes(&[(0, 5)]), free).expect("no geometry, no check");
        assert_eq!((f.events[0].x, f.events[0].y), (1023, 511));
    }

    /// A layout that mirrors a field about an edge the field cannot hold is refused before a byte
    /// is read, and a field that is not mirrored is not.
    ///
    /// A mirrored field stores `edge - coordinate`, so an edge past the field's top would write
    /// column 0 of a 2000-wide sensor as 1999 into ten bits — 975 once truncated — and the file
    /// would decode, to the wrong picture.
    #[test]
    fn a_mirror_about_an_edge_the_field_cannot_hold_is_refused() {
        for (layout, field, value, max) in [
            (Aedat2Layout { width: 1025, ..Aedat2Layout::DAVIS346 }, "width", 1025u64, 1024u64),
            (Aedat2Layout { height: 513, ..Aedat2Layout::DAVIS346 }, "height", 513, 512),
        ] {
            match Aedat2::decode(&aedat2_bytes(&[]), layout) {
                Err(DecodeError::FieldOutOfRange { offset: 0, field: f, value: v, max: m }) => {
                    assert_eq!((f, v, m), (field, value, max));
                }
                other => panic!("a {field} of {value} over a {max}-value field decoded: {other:?}"),
            }
            match Aedat2::encode(&[], layout, &[]) {
                Err(EncodeError::FieldOutOfRange { index: 0, field: f, value: v, max: m }) => {
                    assert_eq!((f, v, m), (field, value, max));
                }
                other => panic!("a {field} of {value} over a {max}-value field encoded: {other:?}"),
            }
        }
        // A full field is the widest edge a mirror can use, and it is accepted at both ends.
        let full = Aedat2Layout { width: 1024, height: 512, ..Aedat2Layout::DAVIS346 };
        let corner = [AerEvent { t: 1, x: 1023, y: 0, polarity: Polarity::On }];
        let bytes = Aedat2::encode(&corner, full, &[]).expect("a 1024-wide mirror fits ten bits");
        assert_eq!(Aedat2::decode(&bytes, full).expect("and decodes").events, corner);
        // An unmirrored field has no edge to hold: its columns past the field are merely
        // unreachable, so a wide stated sensor is not a contradiction.
        let plain = Aedat2Layout { x_invert: false, width: 2000, ..Aedat2Layout::DAVIS346 };
        assert!(Aedat2::decode(&aedat2_bytes(&[]), plain).is_ok());
    }

    /// A coordinate field wider than a `u16` has no representable answer, so the layout is refused
    /// before a byte is read rather than saturating every decoded coordinate at 65535.
    #[test]
    fn a_coordinate_field_wider_than_a_u16_is_refused_before_any_byte_is_read() {
        let wide = Aedat2Layout {
            x_shift: 0,
            x_bits: 17,
            x_invert: false,
            y_shift: 17,
            y_bits: 8,
            y_invert: false,
            p_shift: 25,
            p_on_is_one: true,
            special_mask: 0,
            trigger_mask: 0,
            width: 0,
            height: 0,
            source: "a 17-bit column, which no sensor has and a caller can write",
        };
        assert_eq!(wide.overlap(), 0, "the fields do not overlap; the width is the problem");
        assert!(matches!(
            Aedat2::decode(&aedat2_bytes(&[(0x1_FFFF, 1)]), wide),
            Err(DecodeError::FieldOutOfRange { field: "x_bits", value: 131_071, max: 65_535, .. })
        ));
        assert!(matches!(
            Aedat2::encode(&[], wide, &[]),
            Err(EncodeError::FieldOutOfRange { field: "x_bits", value: 131_071, max: 65_535, .. })
        ));
        // A 16-bit field is the widest that is representable, and it is accepted.
        let ok = Aedat2Layout { x_bits: 16, y_shift: 16, ..wide };
        assert!(Aedat2::decode(&aedat2_bytes(&[]), ok).is_ok());
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

    /// A declared width or height no `u16` coordinate could reach is refused, not folded to 65535.
    ///
    /// The declared geometry is what every column and row below it is range-checked against, so
    /// saturating a declared width of 100,000 to 65,535 — measured, that is what it used to do —
    /// range-checks the whole file against a number that appears in it nowhere.
    #[test]
    fn flat_refuses_a_declared_geometry_no_coordinate_could_reach() {
        let e = synth(4, 3, 64, 64, 10);
        for (at, field) in [(8usize, "width"), (12, "height")] {
            let mut b = Flat::encode(&e, 64, 64).unwrap();
            b[at..at + 4].copy_from_slice(&100_000u32.to_le_bytes());
            match Flat::decode(&b) {
                Err(DecodeError::FieldOutOfRange { offset, field: f, value, max }) => {
                    assert_eq!((offset, f, value, max), (at, field, 100_000, 65_535));
                }
                other => panic!("a declared {field} of 100,000 was accepted: {other:?}"),
            }
        }
        // 65535 itself is representable, so it is accepted and range-checks as itself.
        let mut b = Flat::encode(&e, 64, 64).unwrap();
        b[8..12].copy_from_slice(&u32::from(u16::MAX).to_le_bytes());
        assert_eq!(Flat::decode(&b).expect("decodable").width, u16::MAX);
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
    // AEDAT 4.0: the container DV writes, the compressed payload refuses, and says which.
    // -----------------------------------------------------------------------------------------

    /// An `AEDAT` 4.0 file assembled by hand from the container's layout, NOT by the encoder under
    /// test: the 14-byte version line, the `IOHeader` behind its size, each packet behind its
    /// stream id and size, then the data table.
    fn aedat4_file(io_header: &[u8], packets: &[(i32, &[u8])], table: &[u8]) -> Vec<u8> {
        let mut f = b"#!AER-DAT4.0\r\n".to_vec();
        f.extend_from_slice(&u32::try_from(io_header.len()).unwrap().to_le_bytes());
        f.extend_from_slice(io_header);
        for &(id, payload) in packets {
            f.extend_from_slice(&id.to_le_bytes());
            f.extend_from_slice(&i32::try_from(payload.len()).unwrap().to_le_bytes());
            f.extend_from_slice(payload);
        }
        f.extend_from_slice(table);
        f
    }

    /// The `IOHeader` `DV` wrote into `tests/data/test_data.aedat4` of the `neuromorphicsystems/aedat`
    /// reader, recorded from `DAVIS346_00000002`: its first 48 bytes verbatim — root offset 24,
    /// `IOHE`, six bytes of padding, a 10-byte vtable recording all three fields, and the table —
    /// then an EMPTY `infoNode` string where the recording carries 3,053 bytes of XML. The two
    /// fields a test names are patched in place: compression at byte 28 (1, `LZ4`, in the
    /// recording) and `dataTablePosition` at byte 36 (6,122,043).
    fn dv_io_header(compression: i32, data_table_position: i64) -> Vec<u8> {
        let mut h = vec![
            0x18, 0x00, 0x00, 0x00, 0x49, 0x4f, 0x48, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a,
            0x00, 0x18, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x3b, 0x6a, 0x5d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00,
        ];
        // The emptied infoNode: a length of zero, the NUL FlatBuffers ends a string with, padding.
        h.extend_from_slice(&[0u8; 8]);
        h[28..32].copy_from_slice(&compression.to_le_bytes());
        h[36..44].copy_from_slice(&data_table_position.to_le_bytes());
        h
    }

    /// The event table `DV` wrote, read and written byte for byte.
    ///
    /// The first 64 bytes of the recording's first event packet once `Python`'s `lz4` has
    /// decompressed it: the size prefix, the root offset 16 counted from byte 4, `EVTS`, two bytes
    /// of padding, the vtable, the table, the vector's count, and the first two of the packet's 402
    /// events, verbatim — except that the size prefix (6,460 in the recording) and the count (402)
    /// are set for two events. Through 0.22.0 this reader took the size prefix for the root offset,
    /// and its writer wrote neither a prefix nor an identifier; the two agreed, and neither agreed
    /// with this.
    #[test]
    fn aedat4_reads_and_writes_the_event_table_dv_wrote() {
        let mut dv: [u8; 64] = [
            0x3c, 0x19, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x45, 0x56, 0x54, 0x53, 0x00, 0x00, 0x06,
            0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x92, 0x01,
            0x00, 0x00, 0xa4, 0x99, 0xe3, 0xe0, 0x55, 0xa5, 0x05, 0x00, 0xd7, 0x00, 0xa4, 0x00, 0x01,
            0x00, 0x00, 0x00, 0xf6, 0x99, 0xe3, 0xe0, 0x55, 0xa5, 0x05, 0x00, 0xe2, 0x00, 0xa0, 0x00,
            0x01, 0x00, 0x00, 0x00,
        ];
        assert_eq!(u32::from_le_bytes([dv[0], dv[1], dv[2], dv[3]]), 6_460, "the recording's prefix");
        assert_eq!(u32::from_le_bytes([dv[28], dv[29], dv[30], dv[31]]), 402, "and its count");
        dv[0..4].copy_from_slice(&60u32.to_le_bytes());
        dv[28..32].copy_from_slice(&2u32.to_le_bytes());
        // The two events, read off the bytes by hand: an int64 of microseconds, int16 column, int16
        // row, a bool, three bytes of padding.
        let want = [
            AerEvent { t: 0x0005_A555_E0E3_99A4, x: 215, y: 164, polarity: Polarity::On },
            AerEvent { t: 0x0005_A555_E0E3_99F6, x: 226, y: 160, polarity: Polarity::On },
        ];
        assert_eq!(want[0].t, 1_589_163_147_368_868);
        let mut cursor = 0;
        assert_eq!(Aedat4::read_event_packet(&dv, 0, &mut cursor).expect("DV's own table"), want);
        assert_eq!(cursor, want[1].t, "the cross-packet cursor advanced to the last event");
        // And this module's writer lays the table out exactly as DV did.
        assert_eq!(Aedat4::write_event_packet(&want), dv.to_vec());
        // Through the whole container, as a packet of an uncompressed file.
        let file = aedat4_file(&dv_io_header(0, -1), &[(0, &dv[..])], &[]);
        let d = Aedat4::decode(&file).expect("decodable");
        assert_eq!(d.events().expect("uncompressed events"), want);
        assert_eq!(d.encode().expect("serialisable"), file);
    }

    /// The capability boundary of a zero-dependency crate, asserted rather than described.
    ///
    /// A compressed file still yields its packet framing — stream ids, sizes, offsets — and only
    /// [`Aedat4::events`] refuses, naming the compression and the packet's offset. And the file
    /// survives a decode-encode cycle byte for byte even though nothing read inside it.
    #[test]
    fn aedat4_decodes_the_framing_of_a_compressed_file_and_refuses_its_events() {
        let io = dv_io_header(1, -1);
        let f = aedat4_file(&io, &[(0, &[1u8, 2, 3, 4][..]), (1, &[9u8; 7][..])], &[]);
        let d = Aedat4::decode(&f).expect("the framing decodes whatever the compression");
        assert_eq!(d.compression, Aedat4Compression::Lz4);
        assert_eq!(d.io_header, io);
        assert_eq!(d.packets.len(), 2);
        assert_eq!(d.packets[0].offset, 14 + 4 + io.len());
        assert_eq!(d.packets[1].stream_id, 1);
        assert_eq!(d.packets[1].payload, vec![9u8; 7]);
        assert!(d.packets.iter().all(|p| p.events.is_none() && p.payload_error.is_none()));
        match d.events() {
            Err(DecodeError::UnsupportedCompression { offset, name }) => {
                assert_eq!(name, "LZ4");
                assert_eq!(offset, d.packets[0].offset);
            }
            other => panic!("a compressed payload was not refused: {other:?}"),
        }
        assert_eq!(d.encode().unwrap(), f, "a file we cannot read inside still re-saves exactly");
    }

    /// The version line is the whole header: 14 bytes, then the `IOHeader`'s size.
    ///
    /// Through 0.22.0 the decoder read `#`-prefixed lines after the version line until one read
    /// `#!END-HEADER`. Byte 14 of the `DV` recording is `0x24`, the low byte of its 3,108-byte
    /// `IOHeader` size, so every `DV` file was refused right there. This builds exactly that start
    /// and checks the refusals that remain: another format's first byte, a file cut inside the
    /// line, and an `IOHeader` that is not one.
    #[test]
    fn aedat4_reads_the_version_line_then_the_io_header_and_nothing_between() {
        let mut real_start = b"#!AER-DAT4.0\r\n".to_vec();
        real_start.extend_from_slice(&3_108u32.to_le_bytes());
        assert_eq!(real_start[14], 0x24, "the recording's byte 14");
        let f = aedat4_file(&dv_io_header(0, -1), &[], &[]);
        assert_eq!(&f[..14], b"#!AER-DAT4.0\r\n");
        let d = Aedat4::decode(&f).expect("a version line, an IOHeader and no packets");
        assert!(d.packets.is_empty() && d.data_table.is_empty());
        // An AEDAT 3.1 file differs at the version digit, byte 9, and is named there.
        match Aedat4::decode(b"#!AER-DAT3.1\r\n#!END-HEADER\r\n") {
            Err(DecodeError::BadMagic { offset, expected, found }) => {
                assert_eq!((offset, expected, found.as_str()), (9, "#!AER-DAT4.0\r\n", "#!AER-DAT3.1\r\n"));
            }
            other => panic!("an AEDAT 3.1 file was read as 4.0: {other:?}"),
        }
        // Cut inside the line, a file is short rather than foreign.
        assert!(matches!(
            Aedat4::decode(&f[..9]),
            Err(DecodeError::Truncated { offset: 0, need: 14, have: 9 })
        ));
        assert!(matches!(Aedat4::decode(&[]), Err(DecodeError::Truncated { offset: 0, need: 14, have: 0 })));
        // An IOHeader without its identifier is refused, not defaulted, at the identifier's byte.
        let mut bad = f.clone();
        bad[18 + 4] = b'X';
        match Aedat4::decode(&bad) {
            Err(DecodeError::BadMagic { offset, expected, found }) => {
                assert_eq!((offset, expected, found.as_str()), (22, "IOHE", "XOHE"));
            }
            other => panic!("an IOHeader with the wrong identifier was read: {other:?}"),
        }
    }

    /// The compression is field 0 of the `IOHeader`, an `int32` enum, and `NONE` when absent.
    ///
    /// Through 0.22.0 it was read from a `Format:` header line holding names — `RAW`,
    /// `COMPRESSED_LZ4` — that `AEDAT` 4.0 does not use. Each value below is set in the `IOHeader`
    /// `DV` wrote, at the byte where the recording carries its 1.
    #[test]
    fn the_aedat4_compression_is_the_io_headers_first_field() {
        let cases = [
            (0, Aedat4Compression::None, Some("NONE")),
            (1, Aedat4Compression::Lz4, Some("LZ4")),
            (2, Aedat4Compression::Lz4High, Some("LZ4_HIGH")),
            (3, Aedat4Compression::Zstd, Some("ZSTD")),
            (4, Aedat4Compression::ZstdHigh, Some("ZSTD_HIGH")),
            (9, Aedat4Compression::Other(9), None),
        ];
        for (code, want, name) in cases {
            let f = aedat4_file(&dv_io_header(code, -1), &[], &[]);
            let d = Aedat4::decode(&f).expect("an empty body is a file with no packets");
            assert_eq!(d.compression, want, "{code}");
            assert_eq!((d.compression.code(), d.compression.name()), (code, name));
            assert!(d.packets.is_empty());
            assert_eq!(d.events().expect("no packets, no refusal"), Vec::new());
            assert_eq!(d.encode().expect("serialisable"), f);
        }
        // An undefined value is named by its number when a packet is refused for it.
        let f = aedat4_file(&dv_io_header(9, -1), &[(0, &[0u8; 4][..])], &[]);
        match Aedat4::decode(&f).expect("framing").events() {
            Err(DecodeError::UnsupportedCompression { name, .. }) => assert_eq!(name, "compression type 9"),
            other => panic!("an undefined compression was not refused: {other:?}"),
        }
        // A vtable that does not record the field means the schema's default, NONE — even when
        // the bytes where the field would sit say LZ4. This module's own IOHeader, cut to a
        // 4-byte vtable, with a 1 left in the table.
        let mut h = Aedat4::MINIMAL_IO_HEADER.to_vec();
        h[10] = 4;
        h[20] = 1;
        let d = Aedat4::decode(&aedat4_file(&h, &[], &[])).expect("a vtable with no fields");
        assert_eq!(d.compression, Aedat4Compression::None);
        // And the minimal IOHeader as written records the field, as 0.
        let d = Aedat4::decode(&aedat4_file(&Aedat4::MINIMAL_IO_HEADER, &[], &[])).unwrap();
        assert_eq!(d.compression, Aedat4Compression::None);
    }

    /// A vtable entry of zero is how `FlatBuffers` says a field was not written, so the field reads
    /// as the schema's default whatever the bytes where it would sit hold.
    ///
    /// The `IOHeader` `DV` wrote, with its vtable entry for `compression` (bytes 18-19) and then
    /// for `dataTablePosition` (bytes 20-21) set to zero. The table itself still holds 1, `LZ4`,
    /// at byte 28 and 6,122,043 at byte 36; neither may be read, and the defaults `IOHeader.fbs`
    /// gives are `NONE` and -1.
    #[test]
    fn an_aedat4_vtable_entry_of_zero_reads_as_the_schema_default() {
        let mut h = dv_io_header(1, -1);
        h[18..20].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(h[28], 1, "the table still says LZ4");
        let d = Aedat4::decode(&aedat4_file(&h, &[], &[])).expect("compression not recorded");
        assert_eq!(d.compression, Aedat4Compression::None);
        let mut h = dv_io_header(1, 6_122_043);
        h[20..22].copy_from_slice(&0u16.to_le_bytes());
        let d = Aedat4::decode(&aedat4_file(&h, &[], &[])).expect("no table position recorded");
        assert_eq!(d.compression, Aedat4Compression::Lz4, "the entry beside it is still read");
        assert!(d.data_table.is_empty() && d.packets.is_empty());
    }

    /// Packets stop where the `IOHeader`'s `dataTablePosition` says the file data table starts,
    /// and the table is kept, verbatim, for the re-save.
    ///
    /// "No more data is present after that offset in the file." Through 0.22.0 the table's bytes
    /// were read as packets; here they are the first 16 bytes of the recording's own table, an
    /// `LZ4` frame, whose size field read as a packet's is negative.
    #[test]
    fn aedat4_stops_the_packets_at_the_data_table_the_io_header_names() {
        let table: [u8; 16] = [
            0x04, 0x22, 0x4d, 0x18, 0x40, 0x40, 0xc0, 0x87, 0x56, 0x00, 0x00, 0xd1, 0x1c, 0x90,
            0x00, 0x00,
        ];
        let events = synth(3, 0x7A, 346, 260, 9);
        let packet = Aedat4Packet::from_events(0, &events).expect("encodable");
        let io_len = dv_io_header(0, 0).len();
        let packets_end = 14 + 4 + io_len + 8 + packet.payload.len();
        let at = |p: usize| i64::try_from(p).unwrap();
        let file = aedat4_file(&dv_io_header(0, at(packets_end)), &[(0, &packet.payload)], &table);
        let d = Aedat4::decode(&file).expect("decodable");
        assert_eq!(d.packets.len(), 1);
        assert_eq!(d.data_table, table.to_vec());
        assert_eq!(d.events().expect("uncompressed"), events);
        assert_eq!(d.encode().expect("serialisable"), file, "table and all, byte for byte");
        // The same bytes with the position at -1: the table is read as a packet and refused.
        let unmarked = aedat4_file(&dv_io_header(0, -1), &[(0, &packet.payload)], &table);
        assert!(
            matches!(Aedat4::decode(&unmarked), Err(DecodeError::FieldOutOfRange { field: "packet size", .. })),
            "{:?}",
            Aedat4::decode(&unmarked)
        );
        // A position at the very end is an empty table, which is a file.
        let empty = aedat4_file(&dv_io_header(0, at(packets_end)), &[(0, &packet.payload)], &[]);
        let e = Aedat4::decode(&empty).expect("an empty table at the end");
        assert!(e.data_table.is_empty());
        assert_eq!(e.encode().expect("serialisable"), empty);
        // A position inside the last packet cuts it short.
        let inside = aedat4_file(&dv_io_header(0, at(packets_end - 1)), &[(0, &packet.payload)], &table);
        assert!(matches!(Aedat4::decode(&inside), Err(DecodeError::Truncated { .. })));
        // Positions that are not in the file at all, or are in its header, or are negative
        // without being -1, are refused at the IOHeader.
        let header_end = 14 + 4 + io_len;
        for position in [at(header_end) - 1, 0, at(file.len()) + 1, -2] {
            let f = aedat4_file(&dv_io_header(0, position), &[(0, &packet.payload)], &table);
            assert!(
                matches!(
                    Aedat4::decode(&f),
                    Err(DecodeError::MalformedFlatBuffer { offset: 18, what: "dataTablePosition" })
                ),
                "position {position}: {:?}",
                Aedat4::decode(&f)
            );
        }
        // The first valid position, straight after the IOHeader, is a file with no packets.
        let first = aedat4_file(&dv_io_header(0, at(header_end)), &[], &table);
        let f = Aedat4::decode(&first).expect("a table and no packets");
        assert!(f.packets.is_empty());
        assert_eq!(f.data_table, table.to_vec());
    }

    /// A data table the packets have moved away from is refused at encode, and
    /// [`Aedat4::drop_data_table`] is the way out.
    ///
    /// The table indexes the packets by byte offset, so changing a packet's events makes it stale,
    /// and writing it anyway would put an `IOHeader` on disk that points into the middle of a
    /// packet — a file this crate's own decoder would then cut short.
    #[test]
    fn aedat4_refuses_to_write_a_data_table_where_the_io_header_does_not_point() {
        let events = synth(3, 0x7B, 346, 260, 9);
        let packet = Aedat4Packet::from_events(0, &events).expect("encodable");
        let packets_end = 14 + 4 + dv_io_header(0, 0).len() + 8 + packet.payload.len();
        let table = [0xABu8; 8];
        let position = i64::try_from(packets_end).unwrap();
        let file = aedat4_file(&dv_io_header(0, position), &[(0, &packet.payload)], &table);
        let mut d = Aedat4::decode(&file).expect("decodable");
        // One more event: 16 more bytes, and the table would land 16 bytes past where it says.
        let more = synth(4, 0x7B, 346, 260, 9);
        d.packets[0].set_events(&more).expect("encodable");
        let moved = u64::try_from(packets_end + 16).unwrap();
        assert_eq!(
            d.encode(),
            Err(EncodeError::DataTableMisplaced { declared: Some(u64::try_from(packets_end).unwrap()), packets_end: moved })
        );
        // Dropping the table sets the position to -1 in place and empties the table.
        d.drop_data_table().expect("the IOHeader DV wrote records the field");
        assert!(d.data_table.is_empty());
        assert_eq!(&d.io_header[36..44], &(-1i64).to_le_bytes());
        let back = Aedat4::decode(&d.encode().expect("nothing left to misplace")).expect("decodable");
        assert_eq!(back.events().expect("uncompressed"), more);
        assert!(back.data_table.is_empty());
        // A table with no position declared is refused the same way.
        let mut orphan = Aedat4::raw_from_events(&events, 8, 0).expect("encodable");
        orphan.data_table = table.to_vec();
        let end = u64::try_from(14 + 4 + Aedat4::MINIMAL_IO_HEADER.len() + 8 + packet.payload.len()).unwrap();
        assert_eq!(orphan.encode(), Err(EncodeError::DataTableMisplaced { declared: None, packets_end: end }));
        // Dropping it where the IOHeader never recorded a position leaves the IOHeader alone.
        orphan.drop_data_table().expect("readable");
        assert_eq!(orphan.io_header, Aedat4::MINIMAL_IO_HEADER.to_vec());
        assert!(orphan.encode().is_ok());
        // And an IOHeader this crate cannot read cannot say where a table belongs.
        let mut unreadable = Aedat4::raw_from_events(&events, 8, 0).expect("encodable");
        unreadable.io_header = vec![1, 2, 3];
        assert!(matches!(
            unreadable.encode(),
            Err(EncodeError::FieldOutOfRange { field: "an IOHeader this crate cannot read", value: 3, .. })
        ));
        assert!(unreadable.drop_data_table().is_err());
    }

    /// A payload whose `FlatBuffers` offsets point outside it is refused, not followed.
    #[test]
    fn aedat4_refuses_a_flatbuffer_that_points_outside_itself() {
        let e = synth(8, 11, 64, 64, 5);
        let file = Aedat4::raw_from_events(&e, 8, 0).unwrap();
        let good = file.encode().unwrap();
        let decoded = Aedat4::decode(&good).expect("valid to start with");
        // The payload begins 8 bytes past the packet header; its first four bytes are the size
        // prefix and the next four the FlatBuffers root offset. Point it past the end.
        let root_at = decoded.packets[0].offset + 8 + 4;
        let mut bad = good.clone();
        bad[root_at] = 0xFF;
        bad[root_at + 1] = 0xFF;
        // The FRAMING still decodes — that is what `Aedat4::decode` promises for every file — and
        // the refusal lands on the packet and on `events()`, which is the call that claims to hand
        // back events. Both halves are asserted: a decode that returned Ok with `events: None` and
        // an `events()` that returned an empty list would be the silent failure this replaced.
        let d = Aedat4::decode(&bad).expect("the framing survives a broken payload");
        assert_eq!(d.packets.len(), decoded.packets.len(), "the packet boundaries are still read");
        let mut damaged = decoded.packets[0].payload.clone();
        damaged[4] = 0xFF;
        damaged[5] = 0xFF;
        assert_eq!(d.packets[0].payload, damaged, "the payload is preserved verbatim, damage too");
        assert!(d.packets[0].events.is_none());
        assert!(
            matches!(
                d.packets[0].payload_error,
                Some(DecodeError::MalformedFlatBuffer { .. } | DecodeError::CountMismatch { .. })
            ),
            "{:?}",
            d.packets[0].payload_error
        );
        match d.events() {
            Err(DecodeError::MalformedFlatBuffer { .. } | DecodeError::CountMismatch { .. }) => {}
            Err(other) => panic!("wrong refusal: {other:?}"),
            Ok(_) => panic!("a FlatBuffer pointing outside itself was followed"),
        }
        // And the error `events()` gives is the packet's own, not a manufactured stand-in.
        assert_eq!(d.events().unwrap_err(), d.packets[0].payload_error.clone().unwrap());
    }

    /// A size prefix that disagrees with the payload is refused, and a packet of another stream is
    /// refused by its identifier.
    #[test]
    fn aedat4_checks_the_size_prefix_and_the_identifier_of_an_event_packet() {
        let payload = Aedat4::write_event_packet(&synth(2, 0x51, 32, 32, 4));
        assert_eq!(payload.len(), 64);
        let mut cursor = 0;
        assert!(Aedat4::read_event_packet(&payload, 0, &mut cursor).is_ok());
        for prefix in [59u32, 61, 64] {
            let mut p = payload.clone();
            p[0..4].copy_from_slice(&prefix.to_le_bytes());
            assert_eq!(
                Aedat4::read_event_packet(&p, 100, &mut 0),
                Err(DecodeError::CountMismatch { offset: 100, declared: u64::from(prefix), actual: 60 })
            );
        }
        let mut frame = payload.clone();
        frame[8..12].copy_from_slice(b"FRME");
        assert_eq!(
            Aedat4::read_event_packet(&frame, 100, &mut 0),
            Err(DecodeError::BadMagic { offset: 108, expected: "EVTS", found: "FRME".to_string() })
        );
        // A table that records no element vector is an empty packet, not an error: the vtable
        // shortened to its 4-byte head.
        let mut empty = payload.clone();
        empty[14] = 4;
        assert_eq!(Aedat4::read_event_packet(&empty, 0, &mut 0), Ok(Vec::new()));
        // Too short to hold a prefix at all.
        assert_eq!(
            Aedat4::read_event_packet(&[1, 2, 3], 7, &mut 0),
            Err(DecodeError::MalformedFlatBuffer { offset: 7, what: "size prefix" })
        );
    }

    /// An uncompressed `AEDAT` 4.0 file whose packet is not an event table keeps its framing.
    ///
    /// `DV` interleaves frames, inertial samples and triggers with events, as packets in this same
    /// container, so refusing the whole file when one packet is not events means no real `DV`
    /// recording opens at all — which is the opposite of what [`Aedat4`] claims. Measured before
    /// this was fixed: a well-formed file with one 64-byte non-event payload failed the decode
    /// outright with `MalformedFlatBuffer`. The recording this module was checked against has 472
    /// such packets beside its 236 event packets.
    #[test]
    fn aedat4_keeps_the_framing_of_a_raw_file_whose_packet_is_not_events() {
        let events = synth(24, 0x2B, 64, 64, 30);
        // Stream 1: a frame packet's head, as DV frames it — size prefix, root offset and the
        // identifier FRME — over a table this reader has no business walking.
        let mut frame = vec![0x55u8; 64];
        frame[0..4].copy_from_slice(&60u32.to_le_bytes());
        frame[4..8].copy_from_slice(&16u32.to_le_bytes());
        frame[8..12].copy_from_slice(b"FRME");
        let evt = Aedat4::write_event_packet(&events);
        let f = aedat4_file(&dv_io_header(0, -1), &[(5, &frame[..]), (1, &evt[..])], &[]);

        let d = Aedat4::decode(&f).expect("the framing decodes even though one stream is not events");
        assert_eq!(d.compression, Aedat4Compression::None);
        assert_eq!(d.packets.len(), 2);
        assert_eq!((d.packets[0].stream_id, d.packets[1].stream_id), (5, 1));
        assert_eq!(d.packets[0].payload, frame, "the unreadable payload is kept byte for byte");
        assert!(d.packets[0].events.is_none());
        assert_eq!(
            d.packets[0].payload_error,
            Some(DecodeError::BadMagic {
                offset: d.packets[0].offset + 8 + 8,
                expected: "EVTS",
                found: "FRME".to_string()
            }),
            "and it says why, by the identifier"
        );
        // The event stream is fully decoded beside it.
        assert_eq!(d.packets[1].events.as_deref(), Some(&events[..]));
        assert!(d.packets[1].payload_error.is_none());
        // `events()` is a whole-file call, so it still refuses — with the packet's own reason,
        // not with UnsupportedCompression, because the file is not compressed.
        assert_eq!(d.events().unwrap_err(), d.packets[0].payload_error.clone().unwrap());
        // And the whole file re-saves byte for byte.
        assert_eq!(d.encode().unwrap(), f);
    }

    /// An uncompressed packet laid out differently from this module's own writer re-saves byte for
    /// byte.
    ///
    /// `FlatBuffers` is not a canonical encoding: a conforming writer may leave alignment padding
    /// between the vector slot and the vector. Regenerating the payload from the decoded events
    /// discards it — this 56-byte packet would come back out as 48 — so [`Aedat4::encode`] writes
    /// [`Aedat4Packet::payload`] instead.
    #[test]
    fn aedat4_re_saves_a_raw_packet_it_did_not_lay_out_itself_byte_for_byte() {
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&52u32.to_le_bytes()); //  0.. 4  size prefix
        payload.extend_from_slice(&16u32.to_le_bytes()); //  4.. 8  root, from byte 4: table at 20
        payload.extend_from_slice(b"EVTS"); //                8..12  identifier
        payload.extend_from_slice(&[0u8; 2]); //             12..14  padding
        payload.extend_from_slice(&6u16.to_le_bytes()); //   14..16  vtable length
        payload.extend_from_slice(&8u16.to_le_bytes()); //   16..18  table length
        payload.extend_from_slice(&4u16.to_le_bytes()); //   18..20  field 0 at +4
        payload.extend_from_slice(&6i32.to_le_bytes()); //   20..24  soffset to the vtable
        payload.extend_from_slice(&12u32.to_le_bytes()); //  24..28  slot -> the vector at 36
        payload.extend_from_slice(&[0xEEu8; 8]); //          28..36  eight bytes this reader ignores
        payload.extend_from_slice(&1u32.to_le_bytes()); //   36..40  element count
        payload.extend_from_slice(&77i64.to_le_bytes()); //  40..    the element, still 8-aligned
        payload.extend_from_slice(&3i16.to_le_bytes());
        payload.extend_from_slice(&4i16.to_le_bytes());
        payload.push(1);
        payload.extend_from_slice(&[0u8; 3]);
        assert_eq!(payload.len(), 56, "56 bytes in; this module's own writer would lay out 48");

        let f = aedat4_file(&dv_io_header(0, -1), &[(0, &payload[..])], &[]);
        let d = Aedat4::decode(&f).expect("a conforming layout decodes");
        assert_eq!(
            d.packets[0].events.as_deref(),
            Some(&[AerEvent { t: 77, x: 3, y: 4, polarity: Polarity::On }][..]),
            "the events are read through the indirection, gap and all"
        );
        assert_eq!(
            Aedat4::write_event_packet(&d.packets[0].events.clone().unwrap()).len(),
            48,
            "regenerating the payload would lose the eight bytes at 28..36"
        );
        let re = d.encode().expect("serialisable");
        assert_eq!(re, f, "re-saved {} bytes against the {} that came in", re.len(), f.len());
    }

    /// Changing a packet's events goes through [`Aedat4Packet::set_events`], which rewrites the
    /// payload the encoder will write. The two must not be able to drift apart.
    #[test]
    fn aedat4_set_events_rewrites_the_payload_that_gets_written() {
        let first = synth(6, 0x41, 32, 32, 10);
        let second = synth(9, 0x42, 32, 32, 10);
        let mut p = Aedat4Packet::from_events(3, &first).expect("encodable");
        assert_eq!(p.events.as_deref(), Some(&first[..]));
        p.set_events(&second).expect("encodable");
        let file = Aedat4 { packets: vec![p], ..Aedat4::raw_from_events(&[], 1, 0).expect("empty") };
        let back = Aedat4::decode(&file.encode().expect("serialisable")).expect("decodable");
        assert_eq!(back.events().expect("uncompressed"), second, "the file holds the events set last");
        assert_eq!(back.packets[0].stream_id, 3);
        // And the range refusals ride along rather than being bypassed by the back door.
        let wide = [AerEvent { t: 0, x: 32_768, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Aedat4Packet::from_events(0, &wide),
            Err(EncodeError::FieldOutOfRange { field: "column", .. })
        ));
    }

    /// Offsets that would wrap a 32-bit `usize` are refused rather than followed.
    ///
    /// This crate compiles to `wasm32`, where `usize` is 32 bits, and every offset in a
    /// `FlatBuffers` payload is a `u32` the file chose. `24 + 0xFFFF_FFFF` is 23 on that target:
    /// in bounds, and pointing at a length this reader never wrote. The walker does its
    /// arithmetic in `u64` for exactly that reason, which makes the 32-bit and 64-bit paths the
    /// same path and lets this test stand for both.
    #[test]
    fn aedat4_refuses_offsets_that_would_wrap_a_32_bit_usize() {
        let wrap = |patch: &dyn Fn(&mut Vec<u8>)| {
            let mut payload = vec![0u8; 64];
            payload[0..4].copy_from_slice(&60u32.to_le_bytes());
            payload[4..8].copy_from_slice(&16u32.to_le_bytes());
            payload[8..12].copy_from_slice(b"EVTS");
            payload[14..16].copy_from_slice(&6u16.to_le_bytes());
            payload[16..18].copy_from_slice(&8u16.to_le_bytes());
            payload[18..20].copy_from_slice(&4u16.to_le_bytes());
            payload[20..24].copy_from_slice(&6i32.to_le_bytes());
            payload[24..28].copy_from_slice(&4u32.to_le_bytes());
            payload[28..32].copy_from_slice(&0u32.to_le_bytes());
            patch(&mut payload);
            let f = aedat4_file(&dv_io_header(0, -1), &[(0, &payload[..])], &[]);
            Aedat4::decode(&f).expect("framing").packets[0].payload_error.clone()
        };
        // Unpatched, the buffer is a valid empty event packet.
        assert_eq!(wrap(&|_| {}), None, "the fixture itself must decode, or this proves nothing");
        // 4 + the root offset wraps to 2.
        let e = wrap(&|p| p[4..8].copy_from_slice(&0xFFFF_FFFEu32.to_le_bytes()));
        assert!(matches!(e, Some(DecodeError::MalformedFlatBuffer { .. })), "{e:?}");
        // slot + the vector offset wraps back inside the payload.
        let e = wrap(&|p| p[24..28].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()));
        assert!(matches!(e, Some(DecodeError::MalformedFlatBuffer { what: "vector", .. })), "{e:?}");
        // The table + a vtable field offset of 0xFFFF points far past the payload.
        let e = wrap(&|p| p[18..20].copy_from_slice(&0xFFFFu16.to_le_bytes()));
        assert!(matches!(e, Some(DecodeError::MalformedFlatBuffer { what: "vector", .. })), "{e:?}");
        // count * 16 overflows a 32-bit size calculation.
        let e = wrap(&|p| p[28..32].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()));
        assert!(matches!(e, Some(DecodeError::CountMismatch { declared: 4_294_967_295, .. })), "{e:?}");
    }

    /// A length the container's `int32` size field cannot express is refused, not truncated.
    ///
    /// Saturating it at `i32::MAX` writes a **wrong number** into the file: a length field that
    /// disagrees with the bytes after it, which is the one thing a container's framing must never
    /// do. Called directly because no test can allocate the payload that reaches it.
    #[test]
    fn a_payload_too_long_for_the_size_field_is_refused_not_truncated() {
        assert_eq!(Aedat4::size_field(0, "packet size", 5), Ok(5));
        assert_eq!(
            Aedat4::size_field(0, "packet size", usize::try_from(i32::MAX).unwrap()),
            Ok(i32::MAX)
        );
        let over = usize::try_from(i32::MAX).unwrap() + 1;
        assert!(matches!(
            Aedat4::size_field(7, "packet size", over),
            Err(EncodeError::FieldOutOfRange { index: 7, field: "packet size", max: 2_147_483_647, .. })
        ));
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

    /// Every named constant is the value its format's field table prints, checked where the value
    /// itself is load-bearing rather than only where it is used.
    ///
    /// Six of these had no test naming them at all: an opcode constant that is only ever spelled
    /// as a literal inside a fixture is a constant the fixture cannot disagree with.
    #[test]
    fn the_named_constants_are_the_values_the_field_tables_print() {
        // Opcodes, EVT 3.0's 16-bit table and EVT 2.0's 4-bit one.
        assert_eq!(
            [Evt3::ADDR_Y, Evt3::ADDR_X, Evt3::VECT_BASE_X, Evt3::VECT_12, Evt3::VECT_8],
            [0x0, 0x2, 0x3, 0x4, 0x5]
        );
        assert_eq!(
            [Evt3::TIME_LOW, Evt3::CONTINUED_4, Evt3::TIME_HIGH, Evt3::EXT_TRIGGER],
            [0x6, 0x7, 0x8, 0xA]
        );
        assert_eq!([Evt3::OTHERS, Evt3::CONTINUED_12], [0xE, 0xF]);
        assert_eq!(
            [Evt2::CD_OFF, Evt2::CD_ON, Evt2::TIME_HIGH, Evt2::EXT_TRIGGER],
            [0x0, 0x1, 0x8, 0xA]
        );
        assert_eq!([Evt2::OTHERS, Evt2::CONTINUED], [0xE, 0xF]);
        // An EXT_TRIGGER word built from the constants is read back as a trigger by both formats.
        let s = evt3_words(&[(Evt3::EXT_TRIGGER << 12) | 1]);
        assert_eq!(Evt3::decode(&s).unwrap().markers[0].kind, MarkerKind::ExternalTrigger);
        let w = (u32::from(Evt2::EXT_TRIGGER) << 28) | 1;
        assert_eq!(
            Evt2::decode(&w.to_le_bytes()).unwrap().markers[0].kind,
            MarkerKind::ExternalTrigger
        );

        // Magics, each against the bytes its own encoder writes.
        assert_eq!(Aedat2::MAGIC, "!AER-DAT2");
        let a2 = Aedat2::encode(&[], Aedat2Layout::DVS128, &[]).unwrap();
        assert_eq!(a2, b"#!AER-DAT2.0\r\n");
        assert_eq!(Flat::MAGIC, *b"FMAER-01");
        assert_eq!(&Flat::encode(&[], 0, 0).unwrap()[..8], &Flat::MAGIC);
        // AEDAT 4.0: the version line is dv-processing's AEDAT_VERSION_LENGTH of 14 bytes, and
        // the IOHeader's size follows it directly — no text header, no END-HEADER line.
        assert_eq!(Aedat4::MAGIC, "#!AER-DAT4.0\r\n");
        assert_eq!(Aedat4::MAGIC.len(), 14);
        assert_eq!((Aedat4::IO_HEADER_ID, Aedat4::EVENTS_ID), ("IOHE", "EVTS"));
        let a4 = Aedat4::raw_from_events(&[], 8, 0).unwrap().encode().unwrap();
        assert_eq!(&a4[..14], b"#!AER-DAT4.0\r\n");
        assert_eq!(&a4[14..18], &24u32.to_le_bytes(), "the minimal IOHeader's size");
        assert_eq!(&a4[18 + 4..18 + 8], b"IOHE");
        assert_eq!(a4.len(), 18 + 24, "and nothing else in a file with no packets");

        // Record sizes: 16 bytes per FlatBuffers event on top of the 32-byte head DV's own event
        // packets carry, and 16 bytes per Flat record on top of its 24-byte header.
        assert_eq!(Aedat4::FB_EVENT_SIZE, 16);
        let e = synth(5, 9, 16, 16, 4);
        assert_eq!(Aedat4::write_event_packet(&e).len(), 32 + 5 * Aedat4::FB_EVENT_SIZE);
        assert_eq!(Flat::RECORD_SIZE, 16);
        assert_eq!(Flat::HEADER_SIZE, 24);

        // .dat's 14-bit coordinate fields: the largest value encodes, the next is refused.
        assert_eq!(Dat::MAX_COORD, (1 << 14) - 1);
        let wide = [AerEvent { t: 0, x: 16_383, y: 16_383, polarity: Polarity::On }];
        let bytes = Dat::encode(&wide, &[], Dat::CD_TYPE_CODES[0]).expect("14 bits hold 16383");
        assert_eq!(Dat::decode(&bytes).unwrap().events, wide);
        let past = [AerEvent { t: 0, x: 16_384, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Dat::encode(&past, &[], 0x00),
            Err(EncodeError::FieldOutOfRange { field: "column", value: 16_384, max: 16_383, .. })
        ));
        assert_eq!(Dat::RECORD_SIZE, 8);
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

    /// `EVT` 2.0's 28-bit `TIME_HIGH`: a real 2^34 wrap is counted, a jittered word is refused.
    ///
    /// The field is 28 bits, so a wrap is a backwards jump of nearly 2^28 and anything smaller is
    /// corruption. Measured before the half-range rule was added: a `TIME_HIGH` stepping back by
    /// **one**, from 10 to 9, decoded without error and put the next event 17,179,869,120 us — 4 h
    /// 46 min — after its predecessor. `Evt2::encode` refuses a timestamp past
    /// [`Evt2::MAX_TIME_US`], so no stream this crate writes can reach the wrap branch at all;
    /// this is the hand-built stream that does.
    #[test]
    fn evt2_counts_a_real_2_34_wrap_and_refuses_a_jittered_one() {
        let words = |ws: &[u32]| -> Vec<u8> {
            let mut v = Vec::new();
            for w in ws {
                v.extend_from_slice(&w.to_le_bytes());
            }
            v
        };
        let th = |h: u32| (u32::from(Evt2::TIME_HIGH) << 28) | h;
        let cd = |low: u32| (u32::from(Evt2::CD_ON) << 28) | (low << 22) | (5 << 11) | 6;

        // One step back is corruption, and it is named at the offset of the word that did it.
        match Evt2::decode(&words(&[th(10), cd(1), th(9), cd(1)])) {
            Err(DecodeError::NonMonotonicTimestamp { offset, previous, found }) => {
                assert_eq!((offset, previous, found), (8, 10 << 6, 9 << 6));
            }
            other => panic!("a one-step TIME_HIGH decrease was accepted: {other:?}"),
        }
        // Exactly half the range is still ambiguous, so it is still refused.
        assert!(matches!(
            Evt2::decode(&words(&[th(1 << 27), cd(0), th(0)])),
            Err(DecodeError::NonMonotonicTimestamp { .. })
        ));
        // One more than half is a wrap, and the branch the module doc claims is tested.
        let got = Evt2::decode(&words(&[th((1 << 27) + 1), cd(0), th(0), cd(1)]))
            .expect("more than half the range backwards is the wrap");
        assert_eq!(
            got.events.iter().map(|e| e.t).collect::<Vec<_>>(),
            vec![((1u64 << 27) + 1) << 6, (1u64 << 34) + 1]
        );
        // And the real thing: the counter at its top, then zero.
        let got = Evt2::decode(&words(&[th(0x0FFF_FFFF), cd(5), th(0), cd(1)])).expect("a wrap");
        assert_eq!(
            got.events.iter().map(|e| e.t).collect::<Vec<_>>(),
            vec![Evt2::MAX_TIME_US - 58, (1u64 << 34) + 1]
        );
        assert!(monotonic(&got.events));
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

    /// The N-MNIST record against its README's bit table, on hand-built bytes: `x = 17`,
    /// `y = 22`, polarity ON, `t = 0x123456` microseconds is `[0x11, 0x16, 0x92, 0x34, 0x56]`.
    /// Then a round trip, a partial record, and a timestamp that goes backwards.
    #[test]
    fn nmnist_records_match_the_published_bit_table() {
        let bytes = [0x11u8, 0x16, 0x92, 0x34, 0x56, 0x21, 0x00, 0x12, 0x34, 0x57];
        let d = super::NMnist::decode(&bytes).expect("two records");
        assert_eq!(d.events.len(), 2);
        assert_eq!(d.events[0], super::AerEvent { t: 0x12_3456, x: 17, y: 22, polarity: super::Polarity::On });
        assert_eq!(d.events[1], super::AerEvent { t: 0x12_3457, x: 33, y: 0, polarity: super::Polarity::Off });
        assert_eq!(super::NMnist::encode(&d.events).expect("in range"), bytes.to_vec());
        assert_eq!(super::NMnist::MAX_TIMESTAMP, 8_388_607, "23 bits of microseconds is 8.39 s");
        assert!(matches!(
            super::NMnist::decode(&bytes[..7]),
            Err(super::DecodeError::Truncated { offset: 5, need: 5, have: 2 })
        ));
        let mut back = bytes;
        back[9] = 0x55; // t = 0x123455 < 0x123456
        assert!(matches!(
            super::NMnist::decode(&back),
            Err(super::DecodeError::NonMonotonicTimestamp { offset: 5, previous: 0x12_3456, found: 0x12_3455 })
        ));
        assert!(super::NMnist::decode(&[]).expect("empty is empty").events.is_empty());
        // Out of the format's range: refused rather than truncated.
        let wide = super::AerEvent { t: 0, x: 256, y: 0, polarity: super::Polarity::On };
        assert_eq!(super::NMnist::encode(&[wide]), None);
        let late = super::AerEvent { t: 1 << 23, x: 0, y: 0, polarity: super::Polarity::On };
        assert_eq!(super::NMnist::encode(&[late]), None);
        // The polarity bit does not leak into the timestamp: t = 0x7FFFFF with polarity OFF.
        let top = super::AerEvent { t: 0x7F_FFFF, x: 1, y: 2, polarity: super::Polarity::Off };
        let enc = super::NMnist::encode(&[top]).unwrap();
        assert_eq!(enc[2], 0x7F);
        assert_eq!(super::NMnist::decode(&enc).unwrap().events[0], top);
    }

    // ---- AEDAT 3.1 and DVS128 Gesture ----

    /// A packet written byte by byte from the specification, NOT by the encoder under test.
    fn hand_packet(event_type: u16, size: u32, ts_offset: u32, overflow: u32, body: &[u8], number: u32, valid: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&event_type.to_le_bytes());
        p.extend_from_slice(&7u16.to_le_bytes());
        for w in [size, ts_offset, overflow, number, number, valid] {
            p.extend_from_slice(&w.to_le_bytes());
        }
        p.extend_from_slice(body);
        p
    }

    #[test]
    fn aedat3_decodes_a_file_written_by_hand_from_the_specification() {
        let mut file = Vec::from("#!AER-DAT3.1\r\n# a comment line\r\n#!END-HEADER\r\n".as_bytes());
        // x = 100, y = 27, ON, valid: (100 << 17) | (27 << 2) | 0b11 = 13_107_311; t = 1000.
        // x = 5, y = 127, OFF, valid: (5 << 17) | (127 << 2) | 0b01 = 655_869; t = 1500.
        // The same second event again with its validity bit clear: dropped, counted.
        let mut body = Vec::new();
        for (data, t) in [(13_107_311u32, 1000u32), (655_869, 1500), (655_868, 1600)] {
            body.extend_from_slice(&data.to_le_bytes());
            body.extend_from_slice(&t.to_le_bytes());
        }
        file.extend(hand_packet(1, 8, 4, 0, &body, 3, 2));
        // An IMU6 packet (type 3) of two 36-byte events is stepped over by its declared length, and
        // so is a SPECIAL packet (type 0) — whose events are 8 bytes with the timestamp at offset
        // 4, exactly like a polarity event, and would decode as one if the type were not checked.
        file.extend(hand_packet(3, 36, 32, 0, &[0xAB; 72], 2, 2));
        file.extend(hand_packet(0, 8, 4, 0, &[0xFF, 0xFF, 0xFF, 0x7F, 0x10, 0x27, 0, 0], 1, 1));
        // A later polarity packet with the overflow counter at 1: t = 2³¹ + 5.
        let mut late = Vec::new();
        late.extend_from_slice(&((1u32 << 17) | (2 << 2) | 0b11).to_le_bytes());
        late.extend_from_slice(&5u32.to_le_bytes());
        file.extend(hand_packet(1, 8, 4, 1, &late, 1, 1));
        let got = Aedat3::decode(&file).unwrap();
        assert_eq!(
            got.events,
            vec![
                AerEvent { t: 1000, x: 100, y: 27, polarity: Polarity::On },
                AerEvent { t: 1500, x: 5, y: 127, polarity: Polarity::Off },
                AerEvent { t: (1 << 31) + 5, x: 1, y: 2, polarity: Polarity::On },
            ]
        );
        assert_eq!((got.invalid, got.skipped_packets), (1, 2));
    }

    #[test]
    fn aedat3_round_trips_and_refuses_what_the_format_forbids() {
        let mut rng = Rng::new(61);
        let mut t = 2_147_000_000u64; // just under 2³¹, so the run crosses an overflow boundary
        let events: Vec<AerEvent> = (0..500)
            .map(|_| {
                t += u64::from(rng.below(5000));
                AerEvent { t, x: rng.below(128) as u16, y: rng.below(128) as u16, polarity: if rng.below(2) == 1 { Polarity::On } else { Polarity::Off } }
            })
            .collect();
        assert!(events.last().unwrap().t > 1 << 31 && events[0].t < 1 << 31);
        // The overflow counter changes INSIDE a packet, not between two. (With packets of 64 the
        // crossing fell on event 192 = 3 × 64 by luck, the encoder's split at the boundary was never
        // exercised, and removing it survived the mutation sweep.)
        let crossing = events.iter().position(|e| e.t >= 1 << 31).unwrap();
        assert!(crossing % 50 != 0, "the crossing at event {crossing} falls on a packet boundary");
        let bytes = Aedat3::encode(&events, 50).unwrap();
        // 500 events in packets of 50 would be 10 packets; the split at the crossing makes 11.
        assert_eq!(bytes.len(), Aedat3::MAGIC.len() + Aedat3::END_HEADER.len() + 11 * Aedat3::PACKET_HEADER + 500 * 8);
        let back = Aedat3::decode(&bytes).unwrap();
        assert_eq!(back.events, events);
        assert_eq!((back.invalid, back.skipped_packets), (0, 0));
        assert_eq!(Aedat3::decode(&Aedat3::encode(&[], 8).unwrap()).unwrap().events, vec![]);
        assert_eq!(Aedat3::encode(&events, 0), None);
        assert_eq!(Aedat3::encode(&[AerEvent { t: 0, x: 0x8000, y: 0, polarity: Polarity::On }], 8), None);
        assert_eq!(Aedat3::encode(&[events[1], events[0]], 8), None);
        // Refusals, each on a file that differs from a good one in one field.
        let good = Aedat3::encode(&events[..4], 8).unwrap();
        let header = Aedat3::MAGIC.len() + Aedat3::END_HEADER.len();
        assert!(matches!(Aedat3::decode(b"#!AER-DAT4.0\r\n"), Err(DecodeError::BadMagic { offset: 0, .. })));
        assert!(matches!(Aedat3::decode(Aedat3::MAGIC.as_bytes()), Err(DecodeError::BadMagic { .. })), "no END-HEADER");
        assert!(matches!(Aedat3::decode(&good[..good.len() - 3]), Err(DecodeError::Truncated { .. })));
        assert!(matches!(Aedat3::decode(&good[..header + 10]), Err(DecodeError::Truncated { .. })));
        let patch = |offset: usize, value: u32| {
            let mut f = good.clone();
            f[header + offset..header + offset + 4].copy_from_slice(&value.to_le_bytes());
            Aedat3::decode(&f)
        };
        assert!(matches!(patch(16, 5), Err(DecodeError::CountMismatch { declared: 5, actual: 4, .. })), "capacity ≠ number");
        assert!(matches!(patch(24, 3), Err(DecodeError::CountMismatch { declared: 3, actual: 4, .. })), "eventValid ≠ valid events found");
        assert!(matches!(patch(8, 0), Err(DecodeError::UnsupportedRecordLayout { .. })), "timestamp offset is not 4");
        assert!(matches!(patch(12, 0x8000_0000), Err(DecodeError::FieldOutOfRange { field: "eventTSOverflow", .. })));
        assert!(matches!(patch(28 + 4, 0x8000_0001), Err(DecodeError::FieldOutOfRange { field: "timestamp", .. })));
        assert!(matches!(patch(28 + 8 + 4, 1), Err(DecodeError::NonMonotonicTimestamp { .. })), "the second event before the first");
    }

    #[test]
    fn gesture_labels_cut_a_recording_into_its_gestures() {
        let csv = "class,startTime_usec,endTime_usec\n1,1000,1999\n\n8, 5000 ,5000\r\n11,7000,9000\n";
        let labels = gesture_labels(csv).unwrap();
        assert_eq!(
            labels,
            vec![
                GestureLabel { class: 1, start_us: 1000, end_us: 1999 },
                GestureLabel { class: 8, start_us: 5000, end_us: 5000 },
                GestureLabel { class: 11, start_us: 7000, end_us: 9000 },
            ]
        );
        let events: Vec<AerEvent> = [500u64, 1000, 1500, 1999, 2000, 5000, 5000, 8000, 9001]
            .iter()
            .map(|&t| AerEvent { t, x: 1, y: 1, polarity: Polarity::On })
            .collect();
        // Both ends are included. That is this function's stated choice; other readers differ.
        let cut: Vec<Vec<u64>> = labels.iter().map(|l| gesture_events(&events, l).iter().map(|e| e.t).collect()).collect();
        assert_eq!(cut, vec![vec![1000, 1500, 1999], vec![5000, 5000], vec![8000]]);
        assert!(gesture_events(&events, &GestureLabel { class: 2, start_us: 3000, end_us: 4000 }).is_empty());
        assert!(gesture_events(&[], &labels[0]).is_empty());
        assert!(matches!(gesture_labels("class,start,end\n1,2,3\n"), Err(DecodeError::BadMagic { .. })));
        assert!(matches!(gesture_labels(""), Err(DecodeError::BadMagic { .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n1,2\n"), Err(DecodeError::MalformedFlatBuffer { offset: 34, .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n1,2,x\n"), Err(DecodeError::MalformedFlatBuffer { .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n1,2,-3\n"), Err(DecodeError::MalformedFlatBuffer { .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n0,2,3\n"), Err(DecodeError::FieldOutOfRange { field: "class", .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n12,2,3\n"), Err(DecodeError::FieldOutOfRange { field: "class", value: 12, .. })));
        assert!(matches!(gesture_labels("class,startTime_usec,endTime_usec\n3,9,8\n"), Err(DecodeError::FieldOutOfRange { field: "startTime_usec", .. })));
        assert_eq!(gesture_labels("class,startTime_usec,endTime_usec\n").unwrap(), vec![]);
    }

    // -----------------------------------------------------------------------------------------
    // Repairs for mutation survivors: transcribed geometry and polarity, header and record
    // framing, and the axis a coordinate is checked against.
    // -----------------------------------------------------------------------------------------

    /// The `DAVIS346` preset's array size and its polarity convention, pinned on the wire.
    ///
    /// The suite could not see either. Every existing `DAVIS346` test is an encode-decode round
    /// trip through the same preset, and both of these are **symmetric** under that: a width of
    /// 345 refuses the same column on the way in as on the way out, and an inverted
    /// `p_on_is_one` writes the bit one way and reads it back the same way, so the events come
    /// home unchanged. Only the last column of the real array, and the address bit as a number,
    /// can tell. The sense of the polarity bit is the flag this module says it is least sure of;
    /// what is pinned here is that the preset's stated choice is the one the bytes carry, not
    /// that the choice is right for a physical `DAVIS`.
    #[test]
    fn the_davis346_preset_writes_the_array_and_the_polarity_sense_it_states() {
        assert_eq!(
            (Aedat2Layout::DAVIS346.width, Aedat2Layout::DAVIS346.height),
            (346, 260),
            "Taverni et al. 2018 name a 346 x 260 array"
        );
        // The last pixel of that array encodes and comes back.
        let corner = [AerEvent { t: 1, x: 345, y: 259, polarity: Polarity::On }];
        let bytes = Aedat2::encode(&corner, Aedat2Layout::DAVIS346, &[])
            .expect("column 345 and row 259 are inside a 346 x 260 array");
        assert_eq!(Aedat2::decode(&bytes, Aedat2Layout::DAVIS346).unwrap().events, corner);
        // One past each is refused, which is what fixes the edge rather than merely admitting it.
        let past_column = [AerEvent { t: 1, x: 346, y: 0, polarity: Polarity::On }];
        assert!(matches!(
            Aedat2::encode(&past_column, Aedat2Layout::DAVIS346, &[]),
            Err(EncodeError::FieldOutOfRange { field: "column", value: 346, max: 345, .. })
        ));
        let past_row = [AerEvent { t: 1, x: 0, y: 260, polarity: Polarity::On }];
        assert!(matches!(
            Aedat2::encode(&past_row, Aedat2Layout::DAVIS346, &[]),
            Err(EncodeError::FieldOutOfRange { field: "row", value: 260, max: 259, .. })
        ));
        // Polarity in bit 11, big-endian record: On sets it, Off clears it — iniVation's sub-type
        // 10, "DVS Polarity ON". The pixel is the last column and the last row, whose fields are
        // both zero once mirrored, so the address is the polarity bit alone, 0x00000800.
        assert_eq!(
            [Aedat2Layout::DAVIS346.p_on_is_one, Aedat2Layout::DVS128.p_on_is_one],
            [true, false],
            "the two presets state opposite senses, and each says so on its own constant"
        );
        let on = [AerEvent { t: 0, x: 345, y: 259, polarity: Polarity::On }];
        let on_bytes = Aedat2::encode(&on, Aedat2Layout::DAVIS346, &[]).expect("encodable");
        assert_eq!(&on_bytes[on_bytes.len() - 8..], &[0x00, 0x00, 0x08, 0x00, 0, 0, 0, 0]);
        let off = [AerEvent { t: 0, x: 345, y: 259, polarity: Polarity::Off }];
        let off_bytes = Aedat2::encode(&off, Aedat2Layout::DAVIS346, &[]).expect("encodable");
        assert_eq!(&off_bytes[off_bytes.len() - 8..], &[0, 0, 0, 0, 0, 0, 0, 0]);
        // And the decoder reads the same bit the same way.
        assert_eq!(
            Aedat2::decode(&on_bytes, Aedat2Layout::DAVIS346).unwrap().events[0].polarity,
            Polarity::On
        );
        assert_eq!(
            Aedat2::decode(&off_bytes, Aedat2Layout::DAVIS346).unwrap().events[0].polarity,
            Polarity::Off
        );
    }

    /// A [`Aedat2Layout::special_mask`] bit that lands inside the column, the row or the polarity
    /// field is an overlapping layout.
    ///
    /// The suite could not see it: `an_overlapping_address_layout_is_refused` moves `p_shift`
    /// into the column field, so it exercises only the three coordinate-against-coordinate terms
    /// of [`Aedat2Layout::overlap`]. The three terms involving `special_mask` had no test at all,
    /// and dropping them leaves a layout where every pixel of a whole column becomes a marker —
    /// a short, sorted, entirely plausible event list.
    #[test]
    fn a_special_mask_inside_a_coordinate_field_is_an_overlapping_layout() {
        // DVS128: column in bits 1-7, row in bits 8-14, polarity in bit 0.
        for (mask, what) in [(1u32 << 5, "the column"), (1 << 9, "the row"), (1, "the polarity")] {
            let clash = Aedat2Layout { special_mask: mask, ..Aedat2Layout::DVS128 };
            assert_eq!(clash.overlap(), mask, "a special mask over {what} field is an overlap");
            assert!(
                matches!(
                    Aedat2::decode(&aedat2_bytes(&[(0, 0)]), clash),
                    Err(DecodeError::LayoutFieldsOverlap { mask: m }) if m == mask
                ),
                "a special mask over {what} field decoded"
            );
            assert!(matches!(
                Aedat2::encode(&[], clash, &[]),
                Err(EncodeError::FieldOutOfRange {
                    field: "overlapping bit fields in the layout",
                    ..
                })
            ));
        }
        // The shipped presets put their special bits outside every field, which is what makes
        // the refusals above a check rather than a blanket.
        assert_eq!(Aedat2Layout::DVS128.overlap(), 0);
        assert_eq!(Aedat2Layout::DAVIS346.overlap(), 0);
        let free = Aedat2Layout { special_mask: 0, ..Aedat2Layout::DVS128 };
        assert_eq!(free.overlap(), 0, "a mask of zero disables the check, it does not fail it");
    }

    /// The encodable gap is between **consecutive** events, not between an event and the first.
    ///
    /// The suite could not see it: every wrapping-clock test uses two events, where the previous
    /// event and the first event are the same record. A recording longer than 2^31 microseconds
    /// made of small steps would be refused as one impossible gap, which reads as "this format
    /// cannot hold my recording" when the format holds it exactly. Three events, each one
    /// `MAX_GAP_US` after the last, is the shortest fixture that separates the two readings.
    #[test]
    fn the_encodable_gap_is_measured_between_consecutive_events() {
        let step = Aedat2::MAX_GAP_US;
        assert_eq!(step, Dat::MAX_GAP_US, "both formats carry the same 32-bit counter");
        let events = vec![
            AerEvent { t: 0, x: 1, y: 2, polarity: Polarity::Off },
            AerEvent { t: step, x: 3, y: 4, polarity: Polarity::On },
            AerEvent { t: 2 * step, x: 5, y: 6, polarity: Polarity::Off },
        ];
        // 2 * (2^31 - 1) = 4,294,967,294, still inside the 32-bit counter, and each step is
        // exactly the largest gap the unwrapping rule can recover.
        assert_eq!(2 * step, 4_294_967_294);
        let a2 = Aedat2::encode(&events, Aedat2Layout::DVS128, &[])
            .expect("two gaps of MAX_GAP_US each are two encodable gaps");
        assert_eq!(Aedat2::decode(&a2, Aedat2Layout::DVS128).unwrap().events, events);
        let dat = Dat::encode(&events, &[], Dat::CD_TYPE_CODES[0])
            .expect("two gaps of MAX_GAP_US each are two encodable gaps");
        assert_eq!(Dat::decode(&dat).unwrap().events, events);
        // One microsecond more in a single step is still refused, at the event that made it.
        let mut too_far = events.clone();
        too_far[2].t = step + step + 1;
        too_far[1].t = 0;
        assert!(matches!(
            Aedat2::encode(&too_far, Aedat2Layout::DVS128, &[]),
            Err(EncodeError::GapTooLarge { index: 2, .. })
        ));
    }

    /// A decoded column is range-checked against the width and a decoded row against the height.
    ///
    /// The suite could not see it: both presets state both dimensions, so a check that reads the
    /// wrong field still reads a non-zero one and still compares the column with the width. The
    /// two cases that separate the axes are a layout that states only a height — where the
    /// column check must not fire — and one that states only a width, where it must.
    #[test]
    fn each_coordinate_is_range_checked_against_its_own_axis() {
        // Unmirrored, so that the field on the wire is the coordinate and the stated axes are all
        // that differ between the cases below. (A mirror is about the stated edge, so a layout
        // stating no geometry mirrors about the top of the field instead, and these fixtures
        // would change meaning from one layout to the next.)
        let davis = Aedat2Layout { x_invert: false, y_invert: false, ..Aedat2Layout::DAVIS346 };
        let free = Aedat2Layout { width: 0, height: 0, ..davis };
        let wide = [AerEvent { t: 5, x: 400, y: 10, polarity: Polarity::On }];
        let bytes = Aedat2::encode(&wide, free, &[])
            .expect("a layout stating no geometry checks none, and the 10-bit field holds 400");
        // Height stated, width unstated: nothing bounds the column, so the event decodes.
        let no_width = Aedat2Layout { width: 0, ..davis };
        assert_eq!(
            Aedat2::decode(&bytes, no_width).expect("no width, no column check").events,
            wide
        );
        // Width stated, height unstated: the column is checked, and against the width.
        let no_height = Aedat2Layout { height: 0, ..davis };
        match Aedat2::decode(&bytes, no_height) {
            Err(DecodeError::FieldOutOfRange { field, value, max, .. }) => {
                assert_eq!((field, value, max), ("column", 400, 345));
            }
            other => panic!("column 400 was not checked against the stated width 346: {other:?}"),
        }
        // And the row is checked against the height, not the width: a row past 260 inside a
        // layout whose width would admit it.
        let tall = [AerEvent { t: 5, x: 3, y: 300, polarity: Polarity::On }];
        let tall_bytes = Aedat2::encode(&tall, free, &[]).expect("the 9-bit row field holds 300");
        match Aedat2::decode(&tall_bytes, davis) {
            Err(DecodeError::FieldOutOfRange { field, value, max, .. }) => {
                assert_eq!((field, value, max), ("row", 300, 259));
            }
            other => panic!("row 300 was not checked against the stated height 260: {other:?}"),
        }
    }

    /// `EVT` 2.0 writes one `EVT_TIME_HIGH` word per 64-microsecond window, not one per event.
    ///
    /// The suite could not see it: every other `EVT` 2.0 test decodes what it encodes, and a
    /// stream with a redundant time word ahead of every event decodes to exactly the same
    /// events. Only the byte count says the format's whole point — 4 bytes an event — is still
    /// being met; a time word per event doubles a recording on disk and reads as normal.
    #[test]
    fn evt2_writes_one_time_high_word_per_window_not_one_per_event() {
        let same_window = [
            AerEvent { t: 0, x: 1, y: 2, polarity: Polarity::Off },
            AerEvent { t: 5, x: 3, y: 4, polarity: Polarity::On },
        ];
        let bytes = Evt2::encode(&same_window).expect("encodable");
        assert_eq!(bytes.len(), 3 * 4, "one time word and two events is three 32-bit words");
        let words: Vec<u32> =
            bytes.as_chunks::<4>().0.iter().copied().map(u32::from_le_bytes).collect();
        assert_eq!(words[0] >> 28, u32::from(Evt2::TIME_HIGH));
        assert_eq!(words[1] >> 28, u32::from(Evt2::CD_OFF));
        assert_eq!(words[2] >> 28, u32::from(Evt2::CD_ON));
        assert_eq!(Evt2::decode(&bytes).unwrap().events, same_window);
        // 63 and 64 are on opposite sides of the 6-bit low field, so there the word IS written
        // again — which is what stops this test being satisfied by never writing one.
        let crossing = [
            AerEvent { t: 63, x: 1, y: 2, polarity: Polarity::Off },
            AerEvent { t: 64, x: 3, y: 4, polarity: Polarity::On },
        ];
        let across = Evt2::encode(&crossing).expect("encodable");
        assert_eq!(across.len(), 4 * 4, "a new window costs a second time word");
        assert_eq!(Evt2::decode(&across).unwrap().events, crossing);
    }

    /// An `EVT` 3.0 vector mask bit implying a column past the 11-bit column field is refused.
    ///
    /// The suite could not see it: `evt3_vector_masks_expand_to_exactly_the_bits_set` uses small
    /// bases, and the encoder never writes a base within 12 columns of the end of the field, so
    /// no stream this crate produces reaches the bound. A hand-built stream does, and the bound
    /// itself — 2047, the last column the format can express — is what is pinned here: a bit
    /// past it must be an error rather than an event at a column no `Prophesee` sensor has.
    #[test]
    fn evt3_refuses_a_vector_mask_bit_past_the_column_field() {
        // Base 2040 with mask bit 11 implies column 2051.
        let over = evt3_words(&[
            Evt3::ADDR_Y << 12,
            (Evt3::VECT_BASE_X << 12) | 2040,
            (Evt3::VECT_12 << 12) | 0x800,
        ]);
        match Evt3::decode(&over) {
            Err(DecodeError::FieldOutOfRange { field, value, max, .. }) => {
                assert_eq!((field, value, max), ("column implied by a vector mask bit", 2051, 2047));
            }
            other => panic!("a mask bit implying column 2051 was decoded: {other:?}"),
        }
        // Base 2036 with the same bit implies column 2047, the last one that exists, and decodes.
        let edge = evt3_words(&[
            Evt3::ADDR_Y << 12,
            (Evt3::VECT_BASE_X << 12) | 2036,
            (Evt3::VECT_12 << 12) | 0x800,
        ]);
        let back = Evt3::decode(&edge).expect("column 2047 is inside an 11-bit field");
        assert_eq!(back.events.len(), 1);
        assert_eq!(back.events[0].x, 2047);
    }

    /// The vector path takes only **strictly** increasing columns, because a mask carries a
    /// column once however many events named it.
    ///
    /// The suite could not see it: `synth` and the dense fixture both step the column forward by
    /// at least one, so no existing stream repeats a column inside one microsecond and one row.
    /// A real sensor does, and a run that admitted the repeat would fold two events into one
    /// mask bit — a decode that is shorter than what was encoded, with every remaining event
    /// correct.
    #[test]
    fn evt3_vectorises_only_strictly_increasing_columns_so_a_repeated_column_survives() {
        let events: Vec<AerEvent> = [5u16, 5, 6]
            .iter()
            .map(|&x| AerEvent { t: 9, x, y: 3, polarity: Polarity::On })
            .collect();
        let vect = Evt3::encode(&events, true).expect("encodable");
        assert_eq!(
            Evt3::decode(&vect).unwrap().events,
            events,
            "the vectorised encoding lost a repeated column"
        );
        let plain = Evt3::encode(&events, false).expect("encodable");
        assert_eq!(Evt3::decode(&plain).unwrap().events, events);
        // A longer repeat, where the run is long enough that the vector path is tempting twice.
        let many: Vec<AerEvent> = [1u16, 2, 2, 3, 4, 4, 4, 5]
            .iter()
            .map(|&x| AerEvent { t: 11, x, y: 7, polarity: Polarity::Off })
            .collect();
        assert_eq!(Evt3::decode(&Evt3::encode(&many, true).unwrap()).unwrap().events, many);
    }

    /// The encoder re-bases before writing a mask that could not reach the next column, so no
    /// mask word it writes is empty.
    ///
    /// The suite could not see it: `evt3_rebases_a_vector_run_across_a_gap_wider_than_a_mask`
    /// uses a gap of 29 columns, which forces a re-base under any threshold from 12 to 29, and
    /// its byte count is unchanged by a threshold of 16. A gap that lands the next column 13
    /// past the rolling base is the one that separates them, and it costs a `VECT_12` carrying
    /// no bits at all — two bytes that encode nothing, with the events still correct, which is
    /// why a round-trip test cannot see it either.
    #[test]
    fn evt3_never_writes_a_vector_mask_with_no_bits_set() {
        let events: Vec<AerEvent> = [0u16, 1, 25]
            .iter()
            .map(|&x| AerEvent { t: 0, x, y: 42, polarity: Polarity::Off })
            .collect();
        let vect = Evt3::encode(&events, true).expect("encodable");
        assert_eq!(Evt3::decode(&vect).unwrap().events, events);
        // Word for word: time, row, base 0, a 12-mask carrying columns 0 and 1, then a new base
        // at 25 — NOT a second 12-mask carrying nothing across the gap — and an 8-mask for it.
        assert_eq!(
            vect,
            evt3_words(&[
                Evt3::TIME_HIGH << 12,
                Evt3::TIME_LOW << 12,
                (Evt3::ADDR_Y << 12) | 42,
                Evt3::VECT_BASE_X << 12,
                (Evt3::VECT_12 << 12) | 0b11,
                (Evt3::VECT_BASE_X << 12) | 25,
                (Evt3::VECT_8 << 12) | 0b1,
            ])
        );
        // The invariant behind that fixture, over a stream with gaps of every width up to 20.
        let mut spread = Vec::new();
        let mut x = 0u16;
        for step in 1..=20u16 {
            spread.push(AerEvent { t: 1, x, y: 9, polarity: Polarity::On });
            x += step;
        }
        let wide = Evt3::encode(&spread, true).expect("encodable");
        assert_eq!(Evt3::decode(&wide).unwrap().events, spread);
        for w in wide.as_chunks::<2>().0.iter().copied().map(u16::from_le_bytes) {
            let mask = match w >> 12 {
                Evt3::VECT_12 => w & 0x0FFF,
                Evt3::VECT_8 => w & 0x00FF,
                _ => continue,
            };
            assert_ne!(mask, 0, "an empty mask word spends two bytes and carries no event");
        }
    }

    /// A `% Width n` header line admits column `n - 1` and refuses column `n`.
    ///
    /// The suite could not see it: every `.dat` encode in the suite passes an empty header, so
    /// the declared-geometry branch of the encoder is never taken, and the asymmetry it hides is
    /// the dangerous kind — an encoder that admitted column `n` would write a file **its own
    /// decoder refuses**, since the decoder compares the same column with the same declared
    /// width and rejects at `n`.
    #[test]
    fn dat_refuses_the_column_its_own_declared_width_would_refuse_on_the_way_back() {
        let header = vec!["Width 640".to_string(), "Height 480".to_string()];
        let edge = [AerEvent { t: 0, x: 639, y: 479, polarity: Polarity::On }];
        let bytes = Dat::encode(&edge, &header, Dat::CD_TYPE_CODES[0])
            .expect("639 is the last column of a width of 640");
        let back = Dat::decode(&bytes).expect("and the decoder reads it back");
        assert_eq!((back.width, back.height), (Some(640), Some(480)));
        assert_eq!(back.events, edge);
        for (event, field, value, max) in [
            (AerEvent { t: 0, x: 640, y: 0, polarity: Polarity::On }, "column", 640u64, 639u64),
            (AerEvent { t: 0, x: 0, y: 480, polarity: Polarity::On }, "row", 480, 479),
        ] {
            match Dat::encode(&[event], &header, Dat::CD_TYPE_CODES[0]) {
                Err(EncodeError::FieldOutOfRange { field: f, value: v, max: m, .. }) => {
                    assert_eq!((f, v, m), (field, value, max));
                }
                other => panic!("{field} {value} was written into a file declaring {max}+1: {other:?}"),
            }
        }
    }

    /// `.dat` terminates a header line with a bare newline; `AEDAT` 2.0 terminates one with
    /// `CRLF`. They are different conventions and this module writes each one where it belongs.
    ///
    /// The suite could not see it: `dat_round_trips_bit_exactly` and the field-table test both
    /// encode `.dat` with an **empty** header, so no terminator is written at all, and
    /// [`read_header_lines`] strips an optional `\r` on the way back in, so even a header that
    /// was written would round-trip either way. Only the bytes tell, and a `.dat` reader that
    /// splits on `\n` alone hands its caller a `Width` value with a carriage return stuck to it.
    #[test]
    fn dat_terminates_a_header_line_with_a_bare_newline() {
        let mut want = b"%Width 640\n".to_vec();
        want.extend_from_slice(&[Dat::CD_TYPE_CODES[0], Dat::RECORD_SIZE]);
        let bytes = Dat::encode(&[], &["Width 640".to_string()], Dat::CD_TYPE_CODES[0])
            .expect("encodable");
        assert_eq!(bytes, want);
        assert!(!bytes.contains(&b'\r'), "a .dat header carries no carriage return");
        assert_eq!(Dat::decode(&bytes).unwrap().width, Some(640));
        // jAER's AEDAT 2.0, by contrast, does write CRLF, so this is a convention per format and
        // not a blanket preference for one terminator.
        let a2 = Aedat2::encode(&[], Aedat2Layout::DVS128, &[]).expect("encodable");
        assert_eq!(a2, b"#!AER-DAT2.0\r\n");
    }

    /// The flat header's declared count is compared with the records actually present.
    ///
    /// The suite could not see it: `flat_checks_its_own_header_and_reserved_bytes` truncates four
    /// bytes and appends one, and **both** of those leave a body that is not a multiple of 16 —
    /// so the multiple-of-record-size half of the same condition catches them and the count
    /// comparison is never the reason. A count that disagrees while the body stays a whole number
    /// of records is the case that separates the two halves, and it is exactly what a file
    /// truncated at a record boundary looks like.
    #[test]
    fn the_flat_header_count_is_compared_with_the_records_present() {
        let events = synth(4, 3, 64, 64, 10);
        let good = Flat::encode(&events, 64, 64).expect("encodable");
        assert_eq!(good.len(), Flat::HEADER_SIZE + 4 * Flat::RECORD_SIZE);
        assert_eq!(Flat::decode(&good).expect("the fixture is valid").events, events);
        for declared in [0u64, 3, 5, u64::MAX] {
            let mut b = good.clone();
            b[16..24].copy_from_slice(&declared.to_le_bytes());
            match Flat::decode(&b) {
                Err(DecodeError::CountMismatch { offset, declared: d, actual }) => {
                    assert_eq!((offset, d, actual), (16, declared, 4));
                }
                other => panic!("a header declaring {declared} of 4 records was read: {other:?}"),
            }
        }
        // Dropping whole records, which keeps the body a multiple of the record size.
        let mut short = good.clone();
        short.truncate(Flat::HEADER_SIZE + 2 * Flat::RECORD_SIZE);
        match Flat::decode(&short) {
            Err(DecodeError::CountMismatch { offset, declared, actual }) => {
                assert_eq!((offset, declared, actual), (16, 4, 2));
            }
            other => panic!("a file two records short of its header was read: {other:?}"),
        }
    }

    /// A signed coordinate of -1 in an `AEDAT` 4.0 payload is refused like any other negative.
    ///
    /// The suite could not see it: no test puts a negative coordinate in a `FlatBuffers` event at
    /// all, and -1 is the one that hides best — `0xFFFF` is what an all-ones fill writes, and
    /// `unsigned_abs` turns it into a perfectly ordinary event at column 1. A bound of "below
    /// -1" therefore produces a full, sorted, plausible event list off by one pixel from a file
    /// that should not have decoded.
    #[test]
    fn aedat4_refuses_a_coordinate_of_minus_one_like_any_other_negative() {
        let events = [AerEvent { t: 7, x: 3, y: 4, polarity: Polarity::On }];
        let file = Aedat4::raw_from_events(&events, 8, 0).expect("encodable");
        let good = file.encode().expect("encodable");
        assert_eq!(
            Aedat4::decode(&good).expect("the fixture is valid").events().unwrap(),
            events
        );
        // The single packet is last, its 16-byte element vector starts 32 bytes into the payload.
        let first_event = good.len() - file.packets[0].payload.len() + 32;
        for (at, field) in [(8usize, "column"), (10, "row")] {
            for raw in [0xFFFFu16, 0xFFFE, 0x8000] {
                let mut b = good.clone();
                b[first_event + at..first_event + at + 2].copy_from_slice(&raw.to_le_bytes());
                let decoded = Aedat4::decode(&b).expect("the framing survives a bad payload");
                match decoded.events() {
                    Err(DecodeError::FieldOutOfRange { field: f, value, max, .. }) => {
                        assert_eq!((f, value, max), (field, u64::from((raw as i16).unsigned_abs()), 32_767));
                    }
                    other => panic!("a {field} of {} was accepted: {other:?}", raw as i16),
                }
            }
        }
    }

    /// A request for zero events per packet is read as one, not passed to `chunks`.
    ///
    /// The suite could not see it: every call passes 8. `slice::chunks` **panics** on a chunk
    /// size of zero, so the hole here is not a wrong answer but an abort inside a library call,
    /// reachable from a caller who computed the packet size from something that came out empty.
    #[test]
    fn a_request_for_zero_events_per_packet_is_read_as_one_event_per_packet() {
        let events = synth(5, 21, 32, 32, 3);
        let file = Aedat4::raw_from_events(&events, 0, 9).expect("zero is read as one");
        assert_eq!(file.packets.len(), events.len());
        assert!(file.packets.iter().all(|p| p.stream_id == 9));
        assert!(
            file.packets.iter().all(|p| p.events.as_ref().is_some_and(|e| e.len() == 1)),
            "one event per packet"
        );
        assert_eq!(file.events().expect("all uncompressed"), events);
        let bytes = file.encode().expect("encodable");
        assert_eq!(Aedat4::decode(&bytes).unwrap().events().unwrap(), events);
        // And an empty stream with a zero request is an empty file, not a panic either.
        assert!(Aedat4::raw_from_events(&[], 0, 0).expect("encodable").packets.is_empty());
    }

    /// A refused event is numbered within the whole stream the caller passed, not within the
    /// packet it landed in.
    ///
    /// The suite could not see it: the only `AEDAT` 4.0 encode refusals in the suite come from
    /// single-packet fixtures, where the packet index and the stream index are the same number.
    /// A stream index that silently restarts at every packet boundary points the caller at the
    /// wrong event — event 1 of 8,000 rather than event 2,051 — while the error type, the field
    /// and the value are all correct.
    #[test]
    fn aedat4_numbers_a_refused_event_within_the_whole_stream() {
        for (bad, per_packet) in [(3usize, 2usize), (5, 2), (4, 4), (1, 2)] {
            let mut events = synth(6, 5, 32, 32, 3);
            events[bad].x = 40_000;
            match Aedat4::raw_from_events(&events, per_packet, 0) {
                Err(EncodeError::FieldOutOfRange { index, field, value, max }) => {
                    assert_eq!((index, field, value, max), (bad, "column", 40_000, 32_767));
                }
                other => panic!("event {bad} of {per_packet} per packet: {other:?}"),
            }
        }
    }

    /// A [`TrainMap`] that states a height of zero is refused, including for an empty stream.
    ///
    /// The suite could not see it: with events in hand, a height of zero is caught a second time
    /// by the per-event `e.y >= self.height` test, so dropping the header check changes nothing
    /// a non-empty fixture can see. The empty stream is where the two differ, and it is not a
    /// corner: it is what an input pipeline hands this map before the first event arrives, and a
    /// `Some` there says a geometry with no rows in it is a usable map.
    #[test]
    fn a_train_map_stating_no_height_is_refused_even_with_no_events() {
        let no_height = TrainMap { width: 34, height: 0, tick_us: 1, split_polarity: false };
        assert!(no_height.map(&[]).is_none(), "a height of zero is not a geometry");
        assert!(no_height.map(&[AerEvent { t: 0, x: 0, y: 0, polarity: Polarity::On }]).is_none());
        let no_width = TrainMap { width: 0, height: 34, tick_us: 1, split_polarity: false };
        assert!(no_width.map(&[]).is_none(), "a width of zero is not a geometry");
        let no_tick = TrainMap { width: 34, height: 34, tick_us: 0, split_polarity: false };
        assert!(no_tick.map(&[]).is_none(), "a tick of zero microseconds divides by zero");
        // A complete map does return a train for an empty stream, which is what stops the three
        // refusals above being satisfied by a map that refuses everything.
        let whole = TrainMap { width: 34, height: 34, tick_us: 1, split_polarity: false };
        assert!(whole.map(&[]).is_some(), "an empty stream through a stated geometry is a train");
    }

    /// Two N-MNIST events in the same microsecond are in order, not backwards.
    ///
    /// The suite could not see it: `nmnist_records_match_the_published_bit_table` steps the
    /// timestamp by one between its two records and then checks a strict decrease, so equality
    /// is never presented. A saccade puts many events in one microsecond, so a decoder that
    /// refuses equality refuses the dataset it was written for — and refuses it with
    /// "non-monotonic", which reads as a corrupt file rather than as a wrong comparison.
    #[test]
    fn nmnist_accepts_two_events_in_the_same_microsecond() {
        // Two records, both t = 0x123456: x = 17 / y = 22 / ON, then x = 18 / y = 23 / OFF.
        let bytes = [0x11u8, 0x16, 0x92, 0x34, 0x56, 0x12, 0x17, 0x12, 0x34, 0x56];
        let decoded = super::NMnist::decode(&bytes).expect("equal timestamps are in order");
        assert_eq!(decoded.events.len(), 2);
        assert_eq!(decoded.events[0].t, decoded.events[1].t);
        assert_eq!(decoded.events[0].t, 0x12_3456);
        assert_eq!(decoded.events[0].polarity, Polarity::On);
        assert_eq!(decoded.events[1].polarity, Polarity::Off);
        assert_eq!(super::NMnist::encode(&decoded.events).expect("in range"), bytes.to_vec());
        // One microsecond earlier is still refused, so this pins equality and not the check.
        let mut back = bytes;
        back[9] = 0x55;
        assert!(matches!(
            super::NMnist::decode(&back),
            Err(DecodeError::NonMonotonicTimestamp { previous: 0x12_3456, found: 0x12_3455, .. })
        ));
    }

    /// The DVS128 Gesture dataset's sensor is the 128 x 128 array of the chip that recorded it.
    ///
    /// The suite could not see it: `GESTURE_SENSOR` is a transcribed constant that nothing read.
    /// It is the geometry a caller builds a [`TrainMap`] from, so a wrong array size gives an
    /// input layer of the wrong neuron count — 179,920 rather than 32,768 with the polarity
    /// split — and every event still maps to a plausible address.
    #[test]
    fn the_gesture_sensor_is_the_dvs128_array_that_recorded_it() {
        assert_eq!(super::GESTURE_SENSOR, (128, 128));
        assert_eq!(
            super::GESTURE_SENSOR,
            (Aedat2Layout::DVS128.width, Aedat2Layout::DVS128.height),
            "the dataset was recorded with the chip whose layout this module also carries"
        );
        assert_eq!(super::GESTURE_CLASSES, 11);
        let (width, height) = super::GESTURE_SENSOR;
        let split = TrainMap { width, height, tick_us: 1_000, split_polarity: true };
        assert_eq!(split.address_count(), Some(2 * 128 * 128));
        assert_eq!(split.address_count(), Some(32_768));
        assert!(split.map(&[AerEvent { t: 0, x: 127, y: 127, polarity: Polarity::On }]).is_some());
        assert!(split.map(&[AerEvent { t: 0, x: 128, y: 0, polarity: Polarity::On }]).is_none());
        assert!(split.map(&[AerEvent { t: 0, x: 0, y: 128, polarity: Polarity::On }]).is_none());
    }

    /// A gesture label line with a fourth column is refused, not read as its first three.
    ///
    /// The suite could not see it: it presents a line with **two** fields and lines that do not
    /// parse, but never one with more fields than the header names. A reader that takes the
    /// first three of four silently accepts a file with a different schema — which is what a
    /// dataset revision looks like — and returns labels that cut the recording in the wrong
    /// places with nothing to say so.
    #[test]
    fn a_gesture_label_line_with_a_fourth_column_is_refused() {
        let csv = "class,startTime_usec,endTime_usec\n1,1000,1999,7\n";
        let want = csv.find("1,1000").expect("in the fixture");
        match gesture_labels(csv) {
            Err(DecodeError::MalformedFlatBuffer { offset, what }) => {
                assert_eq!(offset, want);
                assert_eq!(what, "a label line is not three unsigned integers");
            }
            other => panic!("a four-column label line was accepted: {other:?}"),
        }
        // The same line with three fields is the one that parses, so this pins the arity and not
        // the parser.
        let three = "class,startTime_usec,endTime_usec\n1,1000,1999\n";
        assert_eq!(
            gesture_labels(three).expect("three fields parse"),
            vec![GestureLabel { class: 1, start_us: 1000, end_us: 1999 }]
        );
    }

    /// The byte offset a bad label line is reported at counts the newlines before it.
    ///
    /// The suite could not see it: every malformed-line fixture in it puts the bad line
    /// **first**, where the offset comes from the header's own length and the per-line advance
    /// has not run yet. From the second line on, an advance that forgets the terminator drifts
    /// by one byte per line — so the offset of the failure in a 100-line file points 99 bytes
    /// before it, which is a number a reader trusts and cannot check.
    #[test]
    fn a_label_lines_offset_counts_the_line_terminators_before_it() {
        let csv = "class,startTime_usec,endTime_usec\n1,1000,1999\n8,2000,2999\nnot a label\n";
        let want = csv.find("not a label").expect("in the fixture");
        assert_eq!(want, 58);
        match gesture_labels(csv) {
            Err(DecodeError::MalformedFlatBuffer { offset, .. }) => assert_eq!(offset, want),
            other => panic!("the fourth line should not parse: {other:?}"),
        }
        // The same drift, reported through a different error, on a line further down still.
        let bad_class = "class,startTime_usec,endTime_usec\n1,0,1\n2,0,1\n3,0,1\n12,0,1\n";
        let at = bad_class.find("12,0,1").expect("in the fixture");
        match gesture_labels(bad_class) {
            Err(DecodeError::FieldOutOfRange { offset, field, value, .. }) => {
                assert_eq!((offset, field, value), (at, "class", 12));
            }
            other => panic!("class 12 should be out of range: {other:?}"),
        }
    }
}
