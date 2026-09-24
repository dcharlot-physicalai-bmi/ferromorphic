//! A strict JSON reader, so that published model files can be read as published.
//!
//! The Allen Cell Types Database serves every fitted GLIF model as a `neuron_config.json`; NIR
//! graphs, dataset sidecars and hardware descriptions arrive as JSON too. A crate with no
//! dependencies either reads that text itself or asks its user to transcribe numbers by hand —
//! which is how a fitted parameter becomes a typo.
//!
//! # What "strict" means here
//!
//! RFC 8259, and nothing it does not allow:
//!
//! - numbers are `-?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?` and are converted by Rust's
//!   correctly rounded `str::parse::<f64>`, so a value written by Python's shortest-round-trip
//!   `repr` comes back to the same bits;
//! - ⚠ `NaN`, `Infinity` and `-Infinity` are refused. Python's `json.dumps` WRITES them by default
//!   for non-finite floats, so a file that contains them is not JSON, and a reader that quietly
//!   accepts them is how a NaN parameter enters a model as though it had been fitted;
//! - a duplicated key in one object is refused. The RFC leaves its meaning to the implementation,
//!   and implementations disagree — first wins in some, last in others — so a file with one has no
//!   single meaning to read;
//! - strings must be valid: every escape is one the RFC lists, `\u` surrogates come in pairs, and
//!   raw control characters are refused;
//! - nesting deeper than [`MAX_DEPTH`] is refused rather than allowed to exhaust the stack.
//!
//! An object keeps its members in file order.

use core::fmt;

/// The deepest nesting of arrays and objects [`parse`] accepts.
pub const MAX_DEPTH: usize = 128;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, as the nearest `f64`.
    Number(f64),
    /// A string, unescaped.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, members in file order, keys unique.
    Object(Vec<(String, Json)>),
}

/// Why a text is not JSON, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    /// Byte offset into the text at which the reader stopped.
    pub offset: usize,
    /// What it expected or found there.
    pub what: &'static str,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not JSON at byte {}: {}", self.offset, self.what)
    }
}

impl std::error::Error for JsonError {}

impl Json {
    /// The member `key` of an object; `None` for a missing key or a value that is not an object.
    ///
    /// [`parse`] never builds an object with a repeated key, but the variant is public and one can be
    /// built by hand; `get` then returns the FIRST member of that name.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The number, if this is one.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(x) => Some(*x),
            _ => None,
        }
    }

    /// The string, if this is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// Read one JSON value from `text`, which must contain nothing else but whitespace.
///
/// # Errors
///
/// [`JsonError`] with the byte offset at which the text stopped being JSON.
pub fn parse(text: &str) -> Result<Json, JsonError> {
    let mut r = Reader { text, b: text.as_bytes(), at: 0 };
    r.space();
    let v = r.value(0)?;
    r.space();
    if r.at < r.b.len() {
        return Err(r.fail("trailing text after the value"));
    }
    Ok(v)
}

struct Reader<'a> {
    text: &'a str,
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn fail(&self, what: &'static str) -> JsonError {
        JsonError { offset: self.at, what }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.at).copied()
    }

    fn space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.at += 1;
        }
    }

    fn literal(&mut self, word: &'static str, v: Json) -> Result<Json, JsonError> {
        if self.b[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(v)
        } else {
            Err(self.fail("an unknown literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        match self.peek() {
            None => Err(self.fail("the text ended where a value was expected")),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'[' | b'{') if depth == MAX_DEPTH => Err(self.fail("nesting deeper than MAX_DEPTH")),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.fail("a character that cannot start a value")),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.at;
        while let Some(b'0'..=b'9') = self.peek() {
            self.at += 1;
        }
        self.at - start
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.fail("a minus sign with no digits after it")),
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if self.digits() == 0 {
                return Err(self.fail("a decimal point with no digits after it"));
            }
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.at += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.at += 1;
            }
            if self.digits() == 0 {
                return Err(self.fail("an exponent with no digits"));
            }
        }
        // The grammar above is a subset of what `f64::from_str` reads, so the parse cannot fail; it
        // can overflow, to infinity, which is refused.
        let x: f64 = self.text[start..self.at].parse().unwrap_or(f64::INFINITY);
        if x.is_finite() { Ok(Json::Number(x)) } else { Err(JsonError { offset: start, what: "a number too large for f64" }) }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = match self.peek() {
                Some(c @ b'0'..=b'9') => u32::from(c - b'0'),
                Some(c @ b'a'..=b'f') => u32::from(c - b'a') + 10,
                Some(c @ b'A'..=b'F') => u32::from(c - b'A') + 10,
                _ => return Err(self.fail("a \\u escape without four hex digits")),
            };
            v = v * 16 + d;
            self.at += 1;
        }
        Ok(v)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.at += 1; // the opening quote
        let mut out = String::new();
        loop {
            let run = self.at;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.at += 1;
            }
            // The run stops only at ASCII bytes, so it is whole characters of the `&str`.
            out.push_str(&self.text[run..self.at]);
            match self.peek() {
                None => return Err(self.fail("a string that never closes")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let c = self.peek().ok_or_else(|| self.fail("a string that never closes"))?;
                    self.at += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                if !self.b[self.at..].starts_with(b"\\u") {
                                    return Err(self.fail("a high surrogate with no low surrogate after it"));
                                }
                                self.at += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.fail("a high surrogate followed by something else"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.fail("a low surrogate with no high surrogate before it"));
                            } else {
                                hi
                            };
                            // Every code reaching here is a scalar value — surrogates were paired
                            // above — so the replacement character is never used.
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.fail("an escape the RFC does not list")),
                    }
                }
                Some(_) => return Err(self.fail("a raw control character inside a string")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.at += 1;
        let mut items = Vec::new();
        self.space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.space();
            items.push(self.value(depth)?);
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.fail("an array element followed by neither ',' nor ']'")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.at += 1;
        let mut members: Vec<(String, Json)> = Vec::new();
        self.space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.space();
            if self.peek() != Some(b'"') {
                return Err(self.fail("an object key that is not a string"));
            }
            let key_at = self.at;
            let key = self.string()?;
            if members.iter().any(|(k, _)| *k == key) {
                return Err(JsonError { offset: key_at, what: "a key that appears twice in one object" });
            }
            self.space();
            if self.peek() != Some(b':') {
                return Err(self.fail("an object key with no ':' after it"));
            }
            self.at += 1;
            self.space();
            let v = self.value(depth)?;
            members.push((key, v));
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.fail("an object member followed by neither ',' nor '}'")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Json, JsonError, MAX_DEPTH, parse};

    /// Every kind of value, nested, reads back as written.
    #[test]
    fn every_kind_of_value_reads_back() {
        let v = parse(" {\"a\": [1, -2.5e-3, 0, true, false, null], \"b\": {\"c\": \"d\"}, \"e\": []} \n").unwrap();
        assert_eq!(
            v,
            Json::Object(vec![
                (
                    "a".into(),
                    Json::Array(vec![
                        Json::Number(1.0),
                        Json::Number(-2.5e-3),
                        Json::Number(0.0),
                        Json::Bool(true),
                        Json::Bool(false),
                        Json::Null
                    ])
                ),
                ("b".into(), Json::Object(vec![("c".into(), Json::String("d".into()))])),
                ("e".into(), Json::Array(vec![])),
            ])
        );
        assert_eq!(v.get("b").and_then(|b| b.get("c")).and_then(Json::as_str), Some("d"));
        assert_eq!(v.get("a").and_then(Json::as_array).map(<[Json]>::len), Some(6));
        assert_eq!(v.get("a").and_then(Json::as_array).and_then(|a| a[1].as_f64()), Some(-2.5e-3));
        assert_eq!(v.get("zz"), None);
        assert_eq!(v.get(""), None, "a key is matched whole, not as a prefix");
        assert_eq!(Json::Bool(true).as_f64(), None);
        assert_eq!(Json::Object(vec![]).as_array(), None);
        let twice = Json::Object(vec![("k".into(), Json::Number(1.0)), ("k".into(), Json::Number(2.0))]);
        assert_eq!(twice.get("k"), Some(&Json::Number(1.0)), "the first of a hand-built repeat");
        assert_eq!(parse("\r\n\t 1 \r\n"), Ok(Json::Number(1.0)), "the four whitespace bytes");
        assert_eq!(Json::Null.get("a"), None);
        assert_eq!(Json::Null.as_f64(), None);
        assert_eq!(Json::Null.as_str(), None);
        assert_eq!(Json::Null.as_array(), None);
        assert_eq!(parse("{}"), Ok(Json::Object(vec![])));
        assert_eq!(parse("\"\""), Ok(Json::String(String::new())));
    }

    /// Numbers come back to the bit: the Allen Institute's fitted values, written by Python's
    /// shortest-round-trip `repr`, and the extremes of the format.
    #[test]
    fn numbers_come_back_to_the_bit() {
        for x in [
            7.0249928245181e-11,
            711661985.1607032,
            0.0033333333333333335,
            -5.576151112965266e-11,
            5e-324,
            f64::MAX,
            f64::MIN_POSITIVE,
            -0.0,
            123456789012345680.0,
        ] {
            let text = format!("{x:?}");
            let back = parse(&text).unwrap().as_f64().unwrap();
            assert_eq!(back.to_bits(), x.to_bits(), "{text}");
        }
        assert_eq!(parse("1E+2"), Ok(Json::Number(100.0)));
        assert_eq!(parse("1e-2"), Ok(Json::Number(0.01)));
        assert_eq!(parse("-0"), Ok(Json::Number(-0.0)));
        assert_eq!(parse("10"), Ok(Json::Number(10.0)));
    }

    /// The escapes the RFC lists, a surrogate pair and a multi-byte character all unescape.
    #[test]
    fn strings_unescape_exactly() {
        let text = r#""q\" b\\ s\/ \b\f\n\r\t \u00e9 \ud83d\ude00 π""#;
        // The escapes must reach the reader AS escapes. An editor that decodes `\u` sequences on
        // write turned this line into raw "é 😀" once, and the test passed while testing UTF-8
        // passthrough instead of unescaping.
        assert_eq!(text.matches("\\u").count(), 3);
        let v = parse(text).unwrap();
        assert_eq!(v, Json::String("q\" b\\ s/ \u{8}\u{c}\n\r\t é 😀 π".into()));
        assert_eq!(parse(r#""\u00E9""#), Ok(Json::String("é".into())), "upper-case hex");
        assert_eq!(parse(r#""\uFFFF""#), Ok(Json::String("\u{ffff}".into())));
    }

    /// What is not JSON is refused, each at the byte where it stops being JSON and with its reason.
    #[test]
    fn what_is_not_json_is_refused_where_it_stops() {
        let cases: [(&str, usize, &str); 30] = [
            ("NaN", 0, "a character that cannot start a value"),
            ("\x0c1", 0, "a character that cannot start a value"),
            ("Infinity", 0, "a character that cannot start a value"),
            ("-Infinity", 1, "a minus sign with no digits after it"),
            ("[1, NaN]", 4, "a character that cannot start a value"),
            ("01", 1, "trailing text after the value"),
            ("1.", 2, "a decimal point with no digits after it"),
            (".5", 0, "a character that cannot start a value"),
            ("1e", 2, "an exponent with no digits"),
            ("1e+", 3, "an exponent with no digits"),
            ("+1", 0, "a character that cannot start a value"),
            ("1e400", 0, "a number too large for f64"),
            ("-1e400", 0, "a number too large for f64"),
            ("nul", 0, "an unknown literal"),
            ("tru", 0, "an unknown literal"),
            ("fals", 0, "an unknown literal"),
            ("", 0, "the text ended where a value was expected"),
            ("[1,]", 3, "a character that cannot start a value"),
            ("[1 2]", 3, "an array element followed by neither ',' nor ']'"),
            ("{\"a\" 1}", 5, "an object key with no ':' after it"),
            ("{\"a\": 1 \"b\": 2}", 8, "an object member followed by neither ',' nor '}'"),
            ("{a: 1}", 1, "an object key that is not a string"),
            ("{\"a\": 1, \"a\": 2}", 9, "a key that appears twice in one object"),
            ("\"abc", 4, "a string that never closes"),
            ("\"a\u{1}b\"", 2, "a raw control character inside a string"),
            (r#""\x""#, 3, "an escape the RFC does not list"),
            (r#""\u12G4""#, 5, "a \\u escape without four hex digits"),
            (r#""\ud83d x""#, 7, "a high surrogate with no low surrogate after it"),
            (r#""\ud83d\u0041""#, 13, "a high surrogate followed by something else"),
            (r#""\ude00""#, 7, "a low surrogate with no high surrogate before it"),
        ];
        for (text, offset, what) in cases {
            assert_eq!(parse(text), Err(JsonError { offset, what }), "{text:?}");
        }
        assert_eq!(parse("1 x").unwrap_err().to_string(), "not JSON at byte 2: trailing text after the value");
        assert_eq!(parse("\"\\"), Err(JsonError { offset: 2, what: "a string that never closes" }));
        assert_eq!(parse("[1, 2"), Err(JsonError { offset: 5, what: "an array element followed by neither ',' nor ']'" }));
    }

    /// Nesting is bounded: `MAX_DEPTH` levels read, one more is refused rather than recursed into.
    #[test]
    fn nesting_is_bounded() {
        let ok = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok());
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert_eq!(parse(&deep), Err(JsonError { offset: MAX_DEPTH, what: "nesting deeper than MAX_DEPTH" }));
        let objects = format!("{}1{}", "{\"k\":".repeat(MAX_DEPTH + 1), "}".repeat(MAX_DEPTH + 1));
        assert_eq!(parse(&objects).unwrap_err().what, "nesting deeper than MAX_DEPTH");
        assert_eq!(MAX_DEPTH, 128);
    }
}
