//! `NumPy`'s `.npy` files, read as `NumPy` writes them: a format published benchmark series and
//! saved event streams arrive in — `NeuroBench`'s Mackey–Glass series among them.
//!
//! # The format
//!
//! `numpy.lib.format` (NEP 1): the six bytes `\x93NUMPY`, a major and minor version, a header
//! length — two bytes little-endian in version 1, four in versions 2 and 3 — and a header that is
//! the text of a Python dictionary literal with exactly three keys:
//!
//! ```text
//! {'descr': '<f8', 'fortran_order': False, 'shape': (3751,), }
//! ```
//!
//! padded with spaces and a newline so the data after it is aligned, then the array's bytes.
//!
//! ⚠ The format documents the version 1 and 2 header as ASCII; `NumPy` writes it in Latin-1, and
//! reads it back that way. A structured field named `é` is written as the single byte `0xE9` in a
//! version 1.0 file — version 3.0, whose header is UTF-8, is used only for a name Latin-1 cannot
//! encode, such as `π`. This reader decodes as `NumPy` does, so both files read.
//!
//! This reader takes `descr` as either one scalar type — `<f8`, `>i2`, `|b1`, `<u4` and the rest of
//! the booleans, signed and unsigned integers of one to eight bytes and floats of four and eight,
//! each with its byte order stated (`=`, the writing machine's own order, is refused: `NumPy`
//! resolves it before writing, so a file that carries it did not come from `np.save`) — or a
//! STRUCTURED type, a list of `('name', 'type')` pairs, which is how an event stream saved
//! with named `x`, `y`, `t`, `p` fields arrives. It refuses everything else by name: object arrays
//! (which are pickles, and would need a Python interpreter to read), strings, complex numbers,
//! half precision, datetimes, nested and sub-array fields.
//!
//! # What is checked
//!
//! The header's grammar, exactly: a key missing, repeated or unknown is refused, as is a shape
//! with a negative or non-integer dimension. The data's length must be the shape's product times
//! the item size — a truncated file is an error naming both numbers, not a short array.
//! [`Npy::values`] converts every numeric type to `f64`, refusing a 64-bit integer that `f64` cannot
//! hold exactly rather than rounding it.

use core::fmt;

/// Why a `.npy` file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpyError {
    /// The file does not start with `\x93NUMPY`.
    Magic,
    /// A format version this reader does not know.
    Version {
        /// Major version.
        major: u8,
        /// Minor version.
        minor: u8,
    },
    /// The file ends before what it declares.
    Truncated {
        /// What was being read.
        what: &'static str,
        /// Bytes needed.
        need: usize,
        /// Bytes present.
        have: usize,
    },
    /// The header is not the dictionary the format specifies.
    Header {
        /// Byte offset into the header text, as decoded to UTF-8.
        offset: usize,
        /// What went wrong there.
        what: &'static str,
    },
    /// A type this reader does not read, as the header spells it.
    Unsupported {
        /// The `descr` text.
        descr: String,
    },
    /// A field asked for that the array does not have — or any field of an array that has none.
    NoSuchField {
        /// The name asked for.
        name: String,
    },
    /// A structured array read as though it held one number per element.
    Structured,
    /// A 64-bit integer that has no exact `f64`.
    Inexact {
        /// Its index in the flattened array.
        index: usize,
    },
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magic => write!(f, "not a .npy file: it does not start with \\x93NUMPY"),
            Self::Version { major, minor } => write!(f, ".npy format version {major}.{minor} is not one this reader knows"),
            Self::Truncated { what, need, have } => write!(f, "the file ends inside the {what}: {need} bytes needed, {have} present"),
            Self::Header { offset, what } => write!(f, "the header is not the format's dictionary at byte {offset}: {what}"),
            Self::Unsupported { descr } => write!(f, "dtype {descr:?} is not one this reader reads"),
            Self::NoSuchField { name } => write!(f, "the array has no field {name:?}"),
            Self::Structured => write!(f, "a structured array holds fields, not one number per element; read it by field"),
            Self::Inexact { index } => write!(f, "element {index} is an integer f64 cannot hold exactly"),
        }
    }
}

impl std::error::Error for NpyError {}

/// The kind of one scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `b1`: one byte, zero or not.
    Bool,
    /// `i1` to `i8`.
    Int,
    /// `u1` to `u8`.
    Uint,
    /// `f4` or `f8`.
    Float,
}

/// One scalar type: its kind, its size in bytes, and whether it is stored big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scalar {
    /// Bool, signed, unsigned or float.
    pub kind: Kind,
    /// Bytes per value.
    pub size: usize,
    /// Most significant byte first.
    pub big_endian: bool,
}

/// An array's element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    /// One scalar per element.
    Scalar(Scalar),
    /// Named fields, packed in order.
    Fields(Vec<(String, Scalar)>),
}

impl Dtype {
    /// Bytes per element.
    #[must_use]
    pub fn item_size(&self) -> usize {
        match self {
            Self::Scalar(s) => s.size,
            Self::Fields(f) => f.iter().map(|(_, s)| s.size).sum(),
        }
    }
}

/// A `.npy` array.
#[derive(Debug, Clone, PartialEq)]
pub struct Npy {
    /// The format version, `(major, minor)`.
    pub version: (u8, u8),
    /// The element type.
    pub dtype: Dtype,
    /// Whether the data is in column-major order.
    pub fortran_order: bool,
    /// The dimensions; empty for a scalar.
    pub shape: Vec<usize>,
    /// The raw element bytes, `shape`'s product times the item size.
    pub data: Vec<u8>,
}

/// The largest integer every smaller one of which `f64` holds exactly: `2^53`.
const EXACT: u64 = 1 << 53;

fn scalar(descr: &str) -> Result<Scalar, NpyError> {
    let unsupported = || NpyError::Unsupported { descr: descr.to_owned() };
    let bytes = descr.as_bytes();
    if bytes.len() < 2 {
        return Err(unsupported());
    }
    let big_endian = match bytes[0] {
        b'<' | b'|' => false,
        b'>' => true,
        _ => return Err(unsupported()),
    };
    let size: usize = descr[2..].parse().map_err(|_| unsupported())?;
    let kind = match (bytes[1], size) {
        (b'b', 1) => Kind::Bool,
        (b'i', 1 | 2 | 4 | 8) => Kind::Int,
        (b'u', 1 | 2 | 4 | 8) => Kind::Uint,
        (b'f', 4 | 8) => Kind::Float,
        _ => return Err(unsupported()),
    };
    Ok(Scalar { kind, size, big_endian })
}

/// A cursor over the header text.
struct Text<'a> {
    s: &'a [u8],
    at: usize,
}

impl Text<'_> {
    fn fail(&self, what: &'static str) -> NpyError {
        NpyError::Header { offset: self.at, what }
    }

    fn space(&mut self) {
        while self.at < self.s.len() && self.s[self.at] == b' ' {
            self.at += 1;
        }
    }

    fn eat(&mut self, c: u8, what: &'static str) -> Result<(), NpyError> {
        self.space();
        if self.s.get(self.at) == Some(&c) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.fail(what))
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.space();
        self.s.get(self.at).copied()
    }

    /// A Python string literal in single or double quotes, without escapes.
    fn string(&mut self) -> Result<String, NpyError> {
        self.space();
        let quote = match self.s.get(self.at) {
            Some(&q @ (b'\'' | b'"')) => q,
            _ => return Err(self.fail("a string was expected")),
        };
        let start = self.at + 1;
        let end = self.s[start..].iter().position(|&c| c == quote).map(|n| start + n).ok_or_else(|| self.fail("a string that never closes"))?;
        if self.s[start..end].contains(&b'\\') {
            return Err(self.fail("an escape in a string"));
        }
        self.at = end + 1;
        // The text is a decoded `String` and the quotes are ASCII, so the slice between them is
        // whole characters.
        Ok(String::from_utf8_lossy(&self.s[start..end]).into_owned())
    }

    fn word(&mut self, w: &str) -> bool {
        self.space();
        if self.s[self.at..].starts_with(w.as_bytes()) {
            self.at += w.len();
            true
        } else {
            false
        }
    }
}

fn read_descr(t: &mut Text<'_>) -> Result<Dtype, NpyError> {
    if t.peek() != Some(b'[') {
        return Ok(Dtype::Scalar(scalar(&t.string()?)?));
    }
    t.eat(b'[', "a list was expected")?;
    let mut fields: Vec<(String, Scalar)> = Vec::new();
    loop {
        if t.peek() == Some(b']') {
            t.at += 1;
            break;
        }
        t.eat(b'(', "a (name, type) pair was expected")?;
        let name = t.string()?;
        t.eat(b',', "a ',' between a field's name and its type was expected")?;
        let descr = if t.peek() == Some(b'[') { return Err(NpyError::Unsupported { descr: format!("nested field {name:?}") }) } else { t.string()? };
        if t.peek() == Some(b',') {
            return Err(NpyError::Unsupported { descr: format!("sub-array field {name:?}") });
        }
        t.eat(b')', "a ')' closing a field was expected")?;
        if fields.iter().any(|(n, _)| *n == name) {
            return Err(t.fail("a field name that appears twice"));
        }
        fields.push((name, scalar(&descr)?));
        match t.peek() {
            Some(b',') => t.at += 1,
            Some(b']') => {}
            _ => return Err(t.fail("a ',' or ']' after a field was expected")),
        }
    }
    if fields.is_empty() {
        return Err(t.fail("a structured dtype with no fields"));
    }
    Ok(Dtype::Fields(fields))
}

fn read_shape(t: &mut Text<'_>) -> Result<Vec<usize>, NpyError> {
    t.eat(b'(', "a shape tuple was expected")?;
    let mut dims = Vec::new();
    loop {
        match t.peek() {
            Some(b')') => {
                t.at += 1;
                return Ok(dims);
            }
            Some(b'0'..=b'9') => {
                let start = t.at;
                while t.s.get(t.at).is_some_and(u8::is_ascii_digit) {
                    t.at += 1;
                }
                let d: usize = core::str::from_utf8(&t.s[start..t.at]).ok().and_then(|x| x.parse().ok()).ok_or_else(|| t.fail("a dimension too large"))?;
                dims.push(d);
                match t.peek() {
                    Some(b',') => t.at += 1,
                    Some(b')') => {}
                    _ => return Err(t.fail("a ',' or ')' after a dimension was expected")),
                }
            }
            _ => return Err(t.fail("a dimension must be a non-negative integer")),
        }
    }
}

/// Read a `.npy` file's bytes.
///
/// # Errors
///
/// [`NpyError`] naming what is wrong and where; see the module notes for what is checked.
pub fn parse(bytes: &[u8]) -> Result<Npy, NpyError> {
    if !bytes.starts_with(b"\x93NUMPY") {
        return Err(NpyError::Magic);
    }
    let need = |what, need: usize| if bytes.len() < need { Err(NpyError::Truncated { what, need, have: bytes.len() }) } else { Ok(()) };
    need("preamble", 8)?;
    let (major, minor) = (bytes[6], bytes[7]);
    let (len_at, len_size) = match (major, minor) {
        (1, 0) => (8, 2),
        (2 | 3, 0) => (8, 4),
        _ => return Err(NpyError::Version { major, minor }),
    };
    need("header length", len_at + len_size)?;
    let mut header_len = 0usize;
    for (k, &b) in bytes[len_at..len_at + len_size].iter().enumerate() {
        header_len |= usize::from(b) << (8 * k);
    }
    let start = len_at + len_size;
    need("header", start + header_len)?;
    let raw = &bytes[start..start + header_len];
    // Latin-1 maps each byte to the code point of the same value.
    let decoded: String = if major < 3 {
        raw.iter().map(|&b| char::from(b)).collect()
    } else {
        String::from_utf8(raw.to_vec()).map_err(|e| NpyError::Header { offset: e.utf8_error().valid_up_to(), what: "a version 3 header that is not UTF-8" })?
    };
    let mut t = Text { s: decoded.as_bytes(), at: 0 };
    t.eat(b'{', "the header must open with '{'")?;
    let (mut descr, mut fortran, mut shape) = (None, None, None);
    loop {
        if t.peek() == Some(b'}') {
            t.at += 1;
            break;
        }
        let key_at = t.at;
        let key = t.string()?;
        t.eat(b':', "a ':' after a key was expected")?;
        let seen = match key.as_str() {
            "descr" => descr.replace(read_descr(&mut t)?).is_some(),
            "fortran_order" => {
                let v = if t.word("True") {
                    true
                } else if t.word("False") {
                    false
                } else {
                    return Err(t.fail("fortran_order must be True or False"));
                };
                fortran.replace(v).is_some()
            }
            "shape" => shape.replace(read_shape(&mut t)?).is_some(),
            _ => return Err(NpyError::Header { offset: key_at, what: "a key the format does not have" }),
        };
        if seen {
            return Err(NpyError::Header { offset: key_at, what: "a key that appears twice" });
        }
        match t.peek() {
            Some(b',') => t.at += 1,
            Some(b'}') => {}
            _ => return Err(t.fail("a ',' or '}' after a value was expected")),
        }
    }
    if t.s[t.at..].iter().any(|&c| c != b' ' && c != b'\n') {
        return Err(t.fail("text after the dictionary"));
    }
    let (Some(dtype), Some(fortran_order), Some(shape)) = (descr, fortran, shape) else {
        return Err(NpyError::Header { offset: t.at, what: "descr, fortran_order and shape are all required" });
    };
    let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    let size = count.and_then(|n| n.checked_mul(dtype.item_size())).ok_or(NpyError::Header { offset: 0, what: "a shape too large to address" })?;
    let data_at = start + header_len;
    if bytes.len() - data_at != size {
        return Err(NpyError::Truncated { what: "data", need: size, have: bytes.len() - data_at });
    }
    Ok(Npy { version: (major, minor), dtype, fortran_order, shape, data: bytes[data_at..].to_vec() })
}

fn value(s: Scalar, raw: &[u8], index: usize) -> Result<f64, NpyError> {
    let mut b = [0u8; 8];
    if s.big_endian {
        for (k, &x) in raw.iter().rev().enumerate() {
            b[k] = x;
        }
    } else {
        b[..raw.len()].copy_from_slice(raw);
    }
    let bits = u64::from_le_bytes(b);
    Ok(match (s.kind, s.size) {
        (Kind::Bool, _) => f64::from(u8::from(bits != 0)),
        (Kind::Float, 4) => f64::from(f32::from_bits(bits as u32)),
        (Kind::Float, _) => f64::from_bits(bits),
        (Kind::Uint, _) => {
            if bits > EXACT {
                return Err(NpyError::Inexact { index });
            }
            bits as f64
        }
        (Kind::Int, n) => {
            let shift = 64 - 8 * n as u32;
            let v = ((bits << shift) as i64) >> shift;
            if v.unsigned_abs() > EXACT {
                return Err(NpyError::Inexact { index });
            }
            v as f64
        }
    })
}

impl Npy {
    /// Every element as `f64`, in the FILE's order — column-major when [`Npy::fortran_order`] is set —
    /// for a scalar dtype.
    ///
    /// # Errors
    ///
    /// [`NpyError::Structured`] for a structured array (use [`Npy::field`]);
    /// [`NpyError::Inexact`] for a 64-bit integer beyond `2^53`.
    pub fn values(&self) -> Result<Vec<f64>, NpyError> {
        match &self.dtype {
            Dtype::Scalar(s) => self.data.chunks_exact(s.size).enumerate().map(|(i, c)| value(*s, c, i)).collect(),
            Dtype::Fields(_) => Err(NpyError::Structured),
        }
    }

    /// One field of a structured array, as `f64`, element by element.
    ///
    /// # Errors
    ///
    /// [`NpyError::NoSuchField`] for a name the dtype does not have, or a scalar dtype;
    /// [`NpyError::Inexact`] as for [`Npy::values`].
    pub fn field(&self, name: &str) -> Result<Vec<f64>, NpyError> {
        let missing = || NpyError::NoSuchField { name: name.to_owned() };
        let Dtype::Fields(fields) = &self.dtype else { return Err(missing()) };
        let mut offset = 0;
        let mut found = None;
        for (n, s) in fields {
            if n == name {
                found = Some((offset, *s));
                break;
            }
            offset += s.size;
        }
        let (offset, s) = found.ok_or_else(missing)?;
        let item = self.dtype.item_size();
        self.data.chunks_exact(item).enumerate().map(|(i, c)| value(s, &c[offset..offset + s.size], i)).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::{Dtype, Kind, NpyError, Scalar, parse};

    /// `np.save(np.array([1.5, -2.25, 1e-300]))`
    const F8: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 102, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 2, 192, 89, 243, 248, 194, 31, 110, 165, 1];
    /// a 2 × 3 big-endian `int16` array
    const I2_BIG: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 62, 105, 50, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 32, 51, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 1, 255, 254, 0, 3, 128, 0, 127, 255, 0, 0];
    /// `[True, False, True]`
    const BOOL: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 124, 98, 49, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 1, 0, 1];
    /// a zero-dimensional array, shape `()`
    const SCALAR: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 102, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0, 0, 28, 64];
    /// a 2 × 3 `float32` array in Fortran order
    const FORTRAN: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 102, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 84, 114, 117, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 32, 51, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 128, 63, 0, 0, 128, 64, 0, 0, 0, 64, 0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 192, 64];
    /// a structured array of two events: `x` int16, `t` int64, `p` bool
    const EVENTS: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 91, 40, 39, 120, 39, 44, 32, 39, 60, 105, 50, 39, 41, 44, 32, 40, 39, 116, 39, 44, 32, 39, 60, 105, 56, 39, 41, 44, 32, 40, 39, 112, 39, 44, 32, 39, 124, 98, 49, 39, 41, 93, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 3, 0, 232, 3, 0, 0, 0, 0, 0, 0, 1, 252, 255, 208, 7, 0, 0, 0, 0, 0, 0, 0];
    /// `uint64` values `2^53`, `2^53 + 1`, 255
    const U8_EDGE: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 117, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 0, 0, 32, 0, 255, 0, 0, 0, 0, 0, 0, 0];
    /// `int64` values `−2^53`, `−2^53 − 1`
    const I8_EDGE: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 105, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0, 0, 224, 255, 255, 255, 255, 255, 255, 255, 223, 255];
    /// `uint8` values 0, 200, 255
    const U1: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 124, 117, 49, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 200, 255];
    /// Version 2.0: a field name of 70 000 bytes needs the four-byte header length. 70152 bytes.
    const V2_LEN: usize = 70152;
    /// Version 3.0: a field named `π`, which Latin-1 cannot encode, so `NumPy` writes a UTF-8 header.
    const V3: &[u8] = &[147, 78, 85, 77, 80, 89, 3, 0, 116, 0, 0, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 91, 40, 39, 207, 128, 39, 44, 32, 39, 60, 102, 52, 39, 41, 93, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 63, 0, 0, 128, 191];
    /// Version 1.0 with a field named `é`: `NumPy` writes the header in Latin-1, byte `0xE9`.
    const LATIN1: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 91, 40, 39, 233, 39, 44, 32, 39, 60, 102, 52, 39, 41, 93, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 128, 62, 0, 0, 128, 64];
    /// The first 128 bytes of `NeuroBench`'s `mg_17.npy` — its whole header — and its first three values.
    const MG17_HEADER: &[u8] = &[147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60, 102, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51, 55, 53, 49, 44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10];
    const MG17_FIRST: [f64; 3] = [1.2667777302367338, 1.3034007827142546, 1.2823045380633082];
    const MG17_LEN: usize = 30136;

    /// Every scalar type numpy wrote reads back, value for value, in the file's order.
    #[test]
    fn what_numpy_writes_reads_back() {
        let a = parse(F8).unwrap();
        assert_eq!((a.version, a.shape.as_slice(), a.fortran_order), ((1, 0), &[3][..], false));
        assert_eq!(a.dtype, Dtype::Scalar(Scalar { kind: Kind::Float, size: 8, big_endian: false }));
        assert_eq!(a.values(), Ok(vec![1.5, -2.25, 1e-300]));
        let b = parse(I2_BIG).unwrap();
        assert_eq!(b.shape, [2, 3]);
        assert_eq!(b.dtype, Dtype::Scalar(Scalar { kind: Kind::Int, size: 2, big_endian: true }));
        assert_eq!(b.values(), Ok(vec![1.0, -2.0, 3.0, -32768.0, 32767.0, 0.0]));
        assert_eq!(parse(BOOL).unwrap().values(), Ok(vec![1.0, 0.0, 1.0]));
        let s = parse(SCALAR).unwrap();
        assert_eq!((s.shape.len(), s.values()), (0, Ok(vec![7.0])), "shape () holds one element");
        let f = parse(FORTRAN).unwrap();
        assert!(f.fortran_order);
        assert_eq!(f.values(), Ok(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]), "column-major, as stored");
        assert_eq!(parse(U1).unwrap().values(), Ok(vec![0.0, 200.0, 255.0]));
        // A bool is zero or not: a byte of 2, which `NumPy` never writes, is still `True`.
        let mut two = BOOL.to_vec();
        let last = two.len() - 1;
        two[last] = 2;
        assert_eq!(parse(&two).unwrap().values(), Ok(vec![1.0, 0.0, 1.0]));
        // Python's double quotes are quotes too.
        let dq = with_header(1, "{\"descr\": \"<f8\", \"fortran_order\": False, \"shape\": (1,), }", 8);
        assert_eq!(parse(&dq).unwrap().values(), Ok(vec![0.0]));
    }

    /// A structured array — the shape a saved event stream arrives in — reads field by field.
    #[test]
    fn a_structured_array_reads_by_field() {
        let e = parse(EVENTS).unwrap();
        assert_eq!(e.dtype.item_size(), 11);
        assert_eq!(e.field("x"), Ok(vec![3.0, -4.0]));
        assert_eq!(e.field("t"), Ok(vec![1000.0, 2000.0]));
        assert_eq!(e.field("p"), Ok(vec![1.0, 0.0]));
        assert_eq!(e.field("y"), Err(NpyError::NoSuchField { name: "y".into() }));
        assert_eq!(e.values(), Err(NpyError::Structured));
        assert_eq!(parse(F8).unwrap().field("x"), Err(NpyError::NoSuchField { name: "x".into() }));
        let v3 = parse(V3).unwrap();
        assert_eq!(v3.version, (3, 0));
        assert_eq!(v3.field("π"), Ok(vec![0.5, -1.0]));
        let latin = parse(LATIN1).unwrap();
        assert_eq!(latin.version, (1, 0));
        assert!(LATIN1.contains(&0xE9) && !LATIN1.windows(2).any(|w| w == [0xC3, 0xA9]), "é is one Latin-1 byte, not UTF-8");
        assert_eq!(latin.field("é"), Ok(vec![0.25, 4.0]));
    }

    /// A 64-bit integer `f64` cannot hold exactly is refused, not rounded: `2^53` is fine, one more
    /// is not, in either sign.
    #[test]
    fn an_integer_f64_cannot_hold_is_refused() {
        assert_eq!(parse(U8_EDGE).unwrap().values(), Err(NpyError::Inexact { index: 1 }));
        assert_eq!(parse(I8_EDGE).unwrap().values(), Err(NpyError::Inexact { index: 1 }));
        let mut ok = U8_EDGE.to_vec();
        let at = ok.len() - 24 + 8;
        ok[at..at + 8].copy_from_slice(&7u64.to_le_bytes());
        assert_eq!(parse(&ok).unwrap().values(), Ok(vec![9_007_199_254_740_992.0, 7.0, 255.0]));
    }

    /// `NeuroBench`'s Mackey–Glass series, whose header is the one above, reads as numpy reads it.
    #[test]
    fn neurobenchs_file_reads_as_numpy_reads_it() {
        let mut file = MG17_HEADER.to_vec();
        for v in MG17_FIRST {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.resize(MG17_LEN, 0);
        let a = parse(&file).unwrap();
        assert_eq!(a.shape, [3751]);
        assert_eq!(&a.values().unwrap()[..3], &MG17_FIRST);
    }

    /// Version 2.0's four-byte header length, on a header too long for two bytes.
    #[test]
    fn a_version_two_header_is_read() {
        let name = "a".repeat(70_000);
        let mut text = format!("{{'descr': [('{name}', '<f8')], 'fortran_order': False, 'shape': (1,), }}");
        while (12 + text.len() + 1) % 64 != 0 {
            text.push(' ');
        }
        text.push('\n');
        let mut file = b"\x93NUMPY\x02\x00".to_vec();
        file.extend_from_slice(&(text.len() as u32).to_le_bytes());
        file.extend_from_slice(text.as_bytes());
        file.extend_from_slice(&2.5f64.to_le_bytes());
        assert!(file.len() > 65_536 && V2_LEN > 65_536, "too long for a two-byte header length, as numpy's was");
        let a = parse(&file).unwrap();
        assert_eq!((a.version, a.field(&name)), ((2, 0), Ok(vec![2.5])));
    }

    fn with_header(version: u8, text: &str, data: usize) -> Vec<u8> {
        let mut file = vec![0x93, b'N', b'U', b'M', b'P', b'Y', version, 0];
        file.extend_from_slice(&(text.len() as u16).to_le_bytes());
        file.extend_from_slice(text.as_bytes());
        file.resize(file.len() + data, 0);
        file
    }

    /// Everything that is not what numpy writes is refused, naming what and where.
    #[test]
    fn what_is_not_a_npy_file_is_refused() {
        let msg = |b: &[u8]| parse(b).unwrap_err().to_string();
        assert_eq!(msg(b"PK\x03\x04"), "not a .npy file: it does not start with \\x93NUMPY");
        assert_eq!(msg(b"\x93NUMPY\x01"), "the file ends inside the preamble: 8 bytes needed, 7 present");
        assert_eq!(msg(b"\x93NUMPY\x04\x00"), ".npy format version 4.0 is not one this reader knows");
        assert_eq!(msg(b"\x93NUMPY\x01\x01"), ".npy format version 1.1 is not one this reader knows");
        assert_eq!(msg(b"\x93NUMPY\x01\x00\x10"), "the file ends inside the header length: 10 bytes needed, 9 present");
        assert_eq!(msg(b"\x93NUMPY\x01\x00\x10\x00{}"), "the file ends inside the header: 26 bytes needed, 12 present");
        let ok = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
        assert!(parse(&with_header(1, ok, 16)).is_ok());
        assert_eq!(msg(&with_header(1, ok, 15)), "the file ends inside the data: 16 bytes needed, 15 present");
        assert_eq!(msg(&with_header(1, ok, 17)), "the file ends inside the data: 16 bytes needed, 17 present");
        let cases: [(&str, &str); 18] = [
            ("{'descr': '<f8', 'shape': (2,), }", "the header is not the format's dictionary at byte 33: descr, fortran_order and shape are all required"),
            ("{'descr': '<f8', 'descr': '<f8', 'fortran_order': False, 'shape': (2,), }", "the header is not the format's dictionary at byte 17: a key that appears twice"),
            ("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), 'x': 1}", "the header is not the format's dictionary at byte 56: a key the format does not have"),
            ("{'descr': '<f8', 'fortran_order': 0, 'shape': (2,), }", "the header is not the format's dictionary at byte 34: fortran_order must be True or False"),
            ("{'descr': '<f8', 'fortran_order': False, 'shape': (-2,), }", "the header is not the format's dictionary at byte 51: a dimension must be a non-negative integer"),
            ("{'descr': '<f8', 'fortran_order': False, 'shape': (2.5,), }", "the header is not the format's dictionary at byte 52: a ',' or ')' after a dimension was expected"),
            ("{'descr': '<f8', 'fortran_order': False, 'shape': [2], }", "the header is not the format's dictionary at byte 50: a shape tuple was expected"),
            ("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), } x", "the header is not the format's dictionary at byte 57: text after the dictionary"),
            ("{'descr':\n'<f8'}", "the header is not the format's dictionary at byte 9: a string was expected"),
            ("['descr']", "the header is not the format's dictionary at byte 0: the header must open with '{'"),
            ("{descr: '<f8'}", "the header is not the format's dictionary at byte 1: a string was expected"),
            ("{'descr' '<f8'}", "the header is not the format's dictionary at byte 9: a ':' after a key was expected"),
            ("{'descr': '<f8' 'shape': (2,)}", "the header is not the format's dictionary at byte 16: a ',' or '}' after a value was expected"),
            ("{'descr': '<f8", "the header is not the format's dictionary at byte 10: a string that never closes"),
            ("{'descr': '<f\\8'}", "the header is not the format's dictionary at byte 10: an escape in a string"),
            ("{'descr': [('x', '<f8') ('y', '<f8')]}", "the header is not the format's dictionary at byte 24: a ',' or ']' after a field was expected"),
            ("{'descr': [('x', '<f8'), ('x', '<f8')]}", "the header is not the format's dictionary at byte 37: a field name that appears twice"),
            ("{'descr': [], 'fortran_order': False, 'shape': (), }", "the header is not the format's dictionary at byte 12: a structured dtype with no fields"),
        ];
        for (text, want) in cases {
            assert_eq!(msg(&with_header(1, text, 8)), want, "{text}");
        }
        assert_eq!(msg(&with_header(1, "{'descr': [('x' '<f8')]}", 8)), "the header is not the format's dictionary at byte 16: a ',' between a field's name and its type was expected");
        assert_eq!(msg(&with_header(1, "{'descr': [['x', '<f8']]}", 8)), "the header is not the format's dictionary at byte 11: a (name, type) pair was expected");
        assert_eq!(msg(&with_header(1, "{'descr': [('x', '<f8'], }", 8)), "the header is not the format's dictionary at byte 22: a ')' closing a field was expected");
        let mut bad_utf8 = with_header(3, "{'descr': '<f8'}", 8);
        bad_utf8.splice(8..10, [16, 0, 0, 0]);
        bad_utf8[13] = 0xFF;
        assert_eq!(msg(&bad_utf8), "the header is not the format's dictionary at byte 1: a version 3 header that is not UTF-8");
        assert_eq!(msg(&with_header(1, "{'descr': '<f8', 'fortran_order': False, 'shape': (99999999999999999999999,), }", 8)), "the header is not the format's dictionary at byte 74: a dimension too large");
        assert_eq!(msg(&with_header(1, "{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296, 4294967296), }", 8)), "the header is not the format's dictionary at byte 0: a shape too large to address");
    }

    /// Types this reader does not read are named as the header spells them.
    #[test]
    fn unsupported_types_are_named() {
        let msg = |d: &str| parse(&with_header(1, &format!("{{'descr': {d}, 'fortran_order': False, 'shape': (1,), }}"), 16)).unwrap_err().to_string();
        for (d, name) in [
            ("'|O'", "|O"),
            ("'<c16'", "<c16"),
            ("'<U5'", "<U5"),
            ("'<f2'", "<f2"),
            ("'<M8[ns]'", "<M8[ns]"),
            ("'=f8'", "=f8"),
            ("'<i3'", "<i3"),
            ("'|b2'", "|b2"),
            ("'<u16'", "<u16"),
            ("'f8'", "f8"),
            ("'<f'", "<f"),
            ("'f'", "f"),
            ("'<'", "<"),
            ("''", ""),
            ("'xi4'", "xi4"),
        ] {
            assert_eq!(msg(d), format!("dtype {name:?} is not one this reader reads"), "{d}");
        }
        assert_eq!(msg("[('x', [('y', '<f8')])]"), "dtype \"nested field \\\"x\\\"\" is not one this reader reads");
        assert_eq!(msg("[('x', '<f8', (2,))]"), "dtype \"sub-array field \\\"x\\\"\" is not one this reader reads");
        assert_eq!(NpyError::Inexact { index: 4 }.to_string(), "element 4 is an integer f64 cannot hold exactly");
        assert_eq!(NpyError::NoSuchField { name: "y".into() }.to_string(), "the array has no field \"y\"");
        assert_eq!(NpyError::Structured.to_string(), "a structured array holds fields, not one number per element; read it by field");
    }
}
