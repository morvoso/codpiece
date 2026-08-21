//! JSON-constrained decoding: a byte-level automaton that says whether a
//! candidate token can still lead to valid JSON.
//!
//! The usual implementation builds a bitmask over the whole vocabulary at
//! every step, which is 151,936 token simulations per token generated. This
//! does it the other way round: the sampler already knows which tokens carry
//! the probability mass, so only the top candidates are tested, in
//! descending order, and the first one that keeps the document valid wins.
//! A full scan is the fallback for the rare case where none of the top
//! candidates is legal, so the result is identical to masking — the mask is
//! simply evaluated lazily, on the few tokens that could have been chosen.
//!
//! The automaton follows RFC 8259 exactly, including the three rules a
//! hand-written validator usually gets wrong and which the tests below
//! caught here: a sign is allowed after `e`/`E` but nowhere else in a
//! number, `[1,]` and `{"a":1,}` are invalid, and a leading zero may not be
//! followed by another digit. It is conservative — a token is rejected
//! unless every one of its bytes is legal — so a constrained generation
//! cannot emit invalid JSON.

/// JSON's whitespace is exactly space, tab, newline and carriage return
/// (RFC 8259 §2). Rust's `is_ascii_whitespace` also admits form feed, which
/// a constrained run promptly emitted and no parser accepts.
fn ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Where a number is, byte by byte. Split out because JSON's number grammar
/// is the part with the most ways to be subtly wrong.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Num {
    /// after `-`; a digit must follow
    NeedInt,
    /// the integer part is exactly `0`, so no further digit may follow
    LeadZero,
    /// in integer digits, having started 1-9
    Int,
    /// after `.`; a digit must follow
    NeedFrac,
    Frac,
    /// after `e`/`E`; a sign or a digit may follow
    NeedExpSign,
    /// after the exponent's sign; a digit must follow
    NeedExp,
    Exp,
}

impl Num {
    /// Whether a number may legally end in this state.
    fn terminal(self) -> bool {
        matches!(self, Num::LeadZero | Num::Int | Num::Frac | Num::Exp)
    }
}

/// What the document is in the middle of.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pos {
    /// before the root value
    Root,
    /// a value is required here (after `,` in an array, or after `:`)
    Value,
    /// straight after `[`: a value, or `]` for the empty array
    ValueOrClose,
    /// inside a string; `esc` after a backslash, `u` counts hex digits left
    Str { esc: bool, u: u8, key: bool },
    Number(Num),
    /// inside `true` / `false` / `null`, `n` bytes matched so far
    Lit { word: &'static [u8], n: usize },
    /// a complete value was just closed; expect `,`, `}`, `]`, or the end
    After,
    /// after `,` in an object: a key string is required
    Key,
    /// straight after `{`: a key, or `}` for the empty object
    KeyOrClose,
    /// a key string just closed; expect `:`
    Colon,
    /// the root value is complete; only whitespace may follow
    Done,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Nest {
    Object,
    Array,
}

/// Byte-level JSON validator, cloned cheaply to test a candidate token.
#[derive(Clone, Debug)]
pub struct Json {
    pos: Pos,
    nest: Vec<Nest>,
}

impl Default for Json {
    fn default() -> Self {
        Self::new()
    }
}

impl Json {
    pub fn new() -> Self {
        Json { pos: Pos::Root, nest: Vec::new() }
    }

    /// True once a complete JSON value has been emitted, so generation may
    /// stop here. A number is a value only once it is in a state it may end
    /// in — `1.` and `1e` are not documents.
    pub fn complete(&self) -> bool {
        match self.pos {
            Pos::Done => true,
            Pos::Number(n) => n.terminal() && self.nest.is_empty(),
            Pos::After => self.nest.is_empty(),
            _ => false,
        }
    }

    /// Feed one byte. Returns false if it cannot appear here, leaving `self`
    /// unspecified — callers test on a clone.
    fn byte(&mut self, b: u8) -> bool {
        match self.pos {
            Pos::Str { esc: true, key, .. } => {
                if !matches!(b, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u') {
                    return false;
                }
                self.pos = Pos::Str { esc: false, u: if b == b'u' { 4 } else { 0 }, key };
                true
            }
            Pos::Str { esc: false, u, key } if u > 0 => {
                if !b.is_ascii_hexdigit() {
                    return false;
                }
                self.pos = Pos::Str { esc: false, u: u - 1, key };
                true
            }
            Pos::Str { esc: false, key, .. } => match b {
                b'"' => {
                    self.pos = if key { Pos::Colon } else { Pos::After };
                    true
                }
                b'\\' => {
                    self.pos = Pos::Str { esc: true, u: 0, key };
                    true
                }
                0x00..=0x1F => false,
                _ => true,
            },

            Pos::Lit { word, n } => {
                if word.get(n) != Some(&b) {
                    return false;
                }
                self.pos =
                    if n + 1 == word.len() { Pos::After } else { Pos::Lit { word, n: n + 1 } };
                true
            }

            Pos::Number(st) => self.number(st, b),

            Pos::Root | Pos::Value => self.open_value(b),
            Pos::ValueOrClose => {
                if b == b']' {
                    self.close(Nest::Array)
                } else {
                    self.open_value(b)
                }
            }

            Pos::KeyOrClose if b == b'}' => self.close(Nest::Object),
            Pos::Key | Pos::KeyOrClose => match b {
                b'"' => {
                    self.pos = Pos::Str { esc: false, u: 0, key: true };
                    true
                }
                _ => ws(b),
            },

            Pos::Colon => {
                if b == b':' {
                    self.pos = Pos::Value;
                    true
                } else {
                    ws(b)
                }
            }

            Pos::After => match b {
                b',' => match self.nest.last() {
                    // strict: after a comma a value or key is REQUIRED, which
                    // is what makes [1,] and {"a":1,} invalid
                    Some(Nest::Object) => {
                        self.pos = Pos::Key;
                        true
                    }
                    Some(Nest::Array) => {
                        self.pos = Pos::Value;
                        true
                    }
                    None => false,
                },
                b'}' => self.close(Nest::Object),
                b']' => self.close(Nest::Array),
                _ => ws(b),
            },

            Pos::Done => ws(b),
        }
    }

    fn number(&mut self, st: Num, b: u8) -> bool {
        let digit = b.is_ascii_digit();
        let next = match st {
            Num::NeedInt if b == b'0' => Some(Num::LeadZero),
            Num::NeedInt if digit => Some(Num::Int),
            Num::NeedInt => None,
            // a leading zero may be followed by . or e, never another digit
            Num::LeadZero if digit => None,
            Num::Int if digit => Some(Num::Int),
            Num::LeadZero | Num::Int if b == b'.' => Some(Num::NeedFrac),
            Num::LeadZero | Num::Int if b == b'e' || b == b'E' => Some(Num::NeedExpSign),
            Num::NeedFrac if digit => Some(Num::Frac),
            Num::NeedFrac => None,
            Num::Frac if digit => Some(Num::Frac),
            Num::Frac if b == b'e' || b == b'E' => Some(Num::NeedExpSign),
            // the ONLY place a sign is legal inside a number
            Num::NeedExpSign if b == b'+' || b == b'-' => Some(Num::NeedExp),
            Num::NeedExpSign if digit => Some(Num::Exp),
            Num::NeedExpSign => None,
            Num::NeedExp if digit => Some(Num::Exp),
            Num::NeedExp => None,
            Num::Exp if digit => Some(Num::Exp),
            _ => None,
        };
        match next {
            Some(n) => {
                self.pos = Pos::Number(n);
                true
            }
            None if st.terminal() => {
                // the number ended here; re-read this byte as a closer
                self.pos = Pos::After;
                self.byte(b)
            }
            None => false,
        }
    }

    fn open_value(&mut self, b: u8) -> bool {
        match b {
            b'"' => {
                self.pos = Pos::Str { esc: false, u: 0, key: false };
                true
            }
            b'{' => {
                self.nest.push(Nest::Object);
                self.pos = Pos::KeyOrClose;
                true
            }
            b'[' => {
                self.nest.push(Nest::Array);
                self.pos = Pos::ValueOrClose;
                true
            }
            b'-' => {
                self.pos = Pos::Number(Num::NeedInt);
                true
            }
            b'0' => {
                self.pos = Pos::Number(Num::LeadZero);
                true
            }
            b'1'..=b'9' => {
                self.pos = Pos::Number(Num::Int);
                true
            }
            b't' => {
                self.pos = Pos::Lit { word: b"true", n: 1 };
                true
            }
            b'f' => {
                self.pos = Pos::Lit { word: b"false", n: 1 };
                true
            }
            b'n' => {
                self.pos = Pos::Lit { word: b"null", n: 1 };
                true
            }
            _ => ws(b),
        }
    }

    fn close(&mut self, want: Nest) -> bool {
        if self.nest.last() != Some(&want) {
            return false;
        }
        self.nest.pop();
        self.pos = if self.nest.is_empty() { Pos::Done } else { Pos::After };
        true
    }

    /// True when the root value is closed and nothing but whitespace could
    /// follow. Distinct from `complete`, which also holds mid-number: `1` is
    /// a complete document but more digits may still arrive, so generation
    /// must not be cut short there.
    pub fn finished(&self) -> bool {
        matches!(self.pos, Pos::Done) || (matches!(self.pos, Pos::After) && self.nest.is_empty())
    }

    /// Feed a whole token. Returns the resulting state, or None if any byte
    /// of it is illegal here.
    pub fn accept(&self, bytes: &[u8]) -> Option<Json> {
        let mut next = self.clone();
        for &b in bytes {
            if !next.byte(b) {
                return None;
            }
        }
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &str) -> Option<Json> {
        Json::new().accept(s.as_bytes())
    }

    /// Anything serde_json accepts, this must accept and call complete.
    /// Form feed is whitespace to Rust and not to JSON. A constrained run
    /// emitted runs of it, and every parser rejected the result.
    #[test]
    fn form_feed_is_not_json_whitespace() {
        assert!(Json::new().accept(b"{} ").is_some());
        assert!(Json::new().accept(b"{}\t").is_some());
        assert!(Json::new().accept(b"{}\x0c").is_none());
        assert!(Json::new().accept(b"\x0c{}").is_none());
    }

    /// Generation must stop when the root value closes, including a bare
    /// number, or the model is forced to emit whitespace until max_tokens.
    #[test]
    fn finished_covers_a_root_number() {
        assert!(Json::new().accept(b"{}").unwrap().finished());
        assert!(Json::new().accept(b"[1]").unwrap().finished());
        assert!(Json::new().accept(b"\"hi\"").unwrap().finished());
        assert!(Json::new().accept(b"true").unwrap().finished());
        // still mid-number: more digits may follow, so not finished
        assert!(!Json::new().accept(b"12").unwrap().finished());
        // the space ended it
        assert!(Json::new().accept(b"12 ").unwrap().finished());
    }

    /// A run cut short by max_tokens leaves a valid PREFIX, not garbage:
    /// the constraint held for every token that was emitted. This is the
    /// observed shape of the sampled failures at max_tokens=300.
    #[test]
    fn a_truncated_document_is_still_a_valid_prefix() {
        for s in [
            r#"{"city_name": "Aetheria", "population": 245000, "coordinates": {"latitude": 40.7128"#,
            r#"{"name": "Aetheria", "geography": {"timezone": "UTC-05:00", "land_area_km2": 780.5"#,
            r#"{"a": [1, 2, {"b": "unterminated"#,
        ] {
            let st = Json::new().accept(s.as_bytes());
            assert!(st.is_some(), "constraint should have allowed this prefix: {s}");
            assert!(!st.unwrap().finished(), "should not be finished: {s}");
            // and it is indeed not parseable yet, which is the point
            assert!(serde_json::from_str::<serde_json::Value>(s).is_err());
        }
    }

    #[test]
    fn accepts_valid_documents() {
        for doc in [
            r#"{}"#,
            r#"[]"#,
            r#"{"a": 1}"#,
            r#"{"a": [1, 2.5, -3e10, true, false, null]}"#,
            r#"{"nested": {"deep": {"deeper": [{"x": "y"}]}}}"#,
            r#""just a string""#,
            r#"-0.5e-7"#,
            r#"{"esc": "quote \" backslash \\ newline \n unicode é"}"#,
            r#"[[[[[1]]]]]"#,
            r#"{"k": "", "empty_obj": {}, "empty_arr": []}"#,
        ] {
            let st = feed(doc).unwrap_or_else(|| panic!("rejected valid: {doc}"));
            assert!(st.complete(), "not complete: {doc}");
            // and it really is valid, by an independent parser
            serde_json::from_str::<serde_json::Value>(doc).expect("test doc must be valid");
        }
    }

    /// Structural violations must be refused at the offending byte.
    #[test]
    fn rejects_invalid_documents() {
        for doc in [
            r#"{"a": }"#,      // value missing
            r#"{"a" 1}"#,      // colon missing
            r#"{a: 1}"#,       // unquoted key
            r#"[1,]"#,         // trailing comma opens a value that never comes
            r#"{"a": 1]"#,     // wrong closer
            r#"[1, 2}"#,       // wrong closer
            r#"{"a": 01}"#,    // handled as 0 then 1 -> two values, no comma
            r#"tru"#,          // incomplete literal is not complete
            r#"{"a": 1.}"#,    // fraction needs a digit
            r#"{"a": 1e}"#,    // exponent needs a digit
        ] {
            match feed(doc) {
                None => {}
                Some(st) => assert!(!st.complete(), "accepted as complete: {doc}"),
            }
        }
    }

    /// A raw control byte inside a string is illegal; escaped is fine.
    #[test]
    fn control_bytes_must_be_escaped() {
        assert!(feed("\"a\nb\"").is_none());
        assert!(feed(r#""a\nb""#).is_some());
    }

    /// Tokens arrive in arbitrary pieces; state must carry across them
    /// exactly as if the bytes had been fed one at a time.
    #[test]
    fn splitting_into_tokens_changes_nothing() {
        let doc = r#"{"key": [1, {"z": null}], "b": "x"}"#;
        let whole = feed(doc).expect("valid");
        for split in 1..doc.len() {
            let a = Json::new().accept(doc[..split].as_bytes()).expect("prefix");
            let b = a.accept(doc[split..].as_bytes()).expect("suffix");
            assert_eq!(b.complete(), whole.complete(), "split at {split}");
        }
    }

    /// The property that matters: whatever the automaton allows, byte by
    /// byte, is parseable JSON once complete. Walks a wide space of
    /// documents built only from accepted bytes.
    #[test]
    fn anything_it_accepts_parses() {
        const BYTES: &[u8] = b"{}[]\",:0123456789.eE+-truefalsn \\";
        let mut stack = vec![(Json::new(), String::new())];
        let mut checked = 0;
        while let Some((st, text)) = stack.pop() {
            if st.complete() {
                serde_json::from_str::<serde_json::Value>(&text)
                    .unwrap_or_else(|e| panic!("accepted but unparseable: {text:?}: {e}"));
                checked += 1;
                continue;
            }
            if text.len() >= 12 || checked > 4000 {
                continue;
            }
            for &b in BYTES {
                if let Some(next) = st.accept(&[b]) {
                    let mut t = text.clone();
                    t.push(b as char);
                    stack.push((next, t));
                }
            }
        }
        assert!(checked > 100, "only {checked} documents explored");
    }
}
