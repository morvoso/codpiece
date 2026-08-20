//! Byte-level BPE tokenizer, loaded from GGUF metadata.
//!
//! Scope: `tokenizer.ggml.model = "gpt2"` with `pre = "qwen35"` (also accepts
//! `qwen2`, same machinery). Mirrors llama.cpp's llm_tokenizer_bpe:
//!   1. partition text on special tokens (control + user-defined), greedy,
//!      longest-match at equal positions;
//!   2. split each plain fragment with the pretokenizer regex;
//!   3. byte-encode each piece (GPT-2 byte→unicode map) and BPE-merge by rank
//!      (lowest rank first; leftmost wins ties);
//!   4. look up symbol strings in the vocab.
//!
//! Correctness bar: token-identical with `llama-tokenize` (b10423) on real
//! corpora — enforced by the M1 gate, not assumed.

use std::collections::HashMap;

use fancy_regex::Regex;
use tandem_gguf::{GgufFile, Value};

// Qwen3.5 pretokenizer (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35).
// Differs from qwen2 by grouping marks (\p{M}) with letters.
const PRE_QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const PRE_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GGUF token_type values (gguf.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_i(v: i64) -> TokenType {
        match v {
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

#[derive(Debug)]
pub enum TokError {
    MissingKey(&'static str),
    Unsupported(String),
    Malformed(String),
}

impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokError::MissingKey(k) => write!(f, "gguf missing {k}"),
            TokError::Unsupported(s) => write!(f, "unsupported tokenizer: {s}"),
            TokError::Malformed(s) => write!(f, "malformed tokenizer data: {s}"),
        }
    }
}

impl std::error::Error for TokError {}

pub struct Tokenizer {
    tokens: Vec<String>,
    types: Vec<TokenType>,
    id_of: HashMap<String, u32>,
    /// BPE merge ranks, keyed "left\0right" (NUL never occurs in symbols).
    ranks: HashMap<String, u32>,
    /// Special tokens (control + user-defined) sorted longest-first.
    specials: Vec<(String, u32)>,
    pre: Regex,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

/// GPT-2 byte→unicode map: printable latin-1 ranges map to themselves,
/// everything else to U+0100+n in order.
fn build_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let keep = |b: u16| {
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0u16..256 {
        let c = if keep(b) {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(0x100 + n).unwrap();
            n += 1;
            c
        };
        enc[b as usize] = c;
        dec.insert(c, b as u8);
    }
    (enc, dec)
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Tokenizer, TokError> {
        let model = g
            .kv("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .ok_or(TokError::MissingKey("tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(TokError::Unsupported(format!("model {model:?}")));
        }
        let pre_name = g
            .kv("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .unwrap_or("qwen2");
        let pattern = match pre_name {
            "qwen35" => PRE_QWEN35,
            "qwen2" | "deepseek-r1-qwen" => PRE_QWEN2,
            other => return Err(TokError::Unsupported(format!("pre {other:?}"))),
        };
        let pre = Regex::new(pattern).map_err(|e| TokError::Malformed(e.to_string()))?;

        let tok_vals = g
            .kv("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or(TokError::MissingKey("tokenizer.ggml.tokens"))?;
        let mut tokens = Vec::with_capacity(tok_vals.len());
        for v in tok_vals {
            tokens.push(
                v.as_str()
                    .ok_or_else(|| TokError::Malformed("non-string token".into()))?
                    .to_string(),
            );
        }

        let types: Vec<TokenType> = match g.kv("tokenizer.ggml.token_type").and_then(Value::as_array) {
            Some(arr) => arr
                .iter()
                .map(|v| TokenType::from_i(v.as_u64().map(|u| u as i64).unwrap_or(1)))
                .collect(),
            None => vec![TokenType::Normal; tokens.len()],
        };
        if types.len() != tokens.len() {
            return Err(TokError::Malformed("token_type length mismatch".into()));
        }

        let mut id_of = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            id_of.insert(t.clone(), i as u32);
        }

        let merge_vals = g
            .kv("tokenizer.ggml.merges")
            .and_then(Value::as_array)
            .ok_or(TokError::MissingKey("tokenizer.ggml.merges"))?;
        let mut ranks = HashMap::with_capacity(merge_vals.len());
        for (rank, m) in merge_vals.iter().enumerate() {
            let m = m
                .as_str()
                .ok_or_else(|| TokError::Malformed("non-string merge".into()))?;
            // merges are "left right"; left may not contain a space, right may
            // (llama.cpp splits on the FIRST space).
            let (l, r) = m
                .split_once(' ')
                .ok_or_else(|| TokError::Malformed(format!("merge without space: {m:?}")))?;
            ranks.insert(format!("{l}\0{r}"), rank as u32);
        }

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .zip(types.iter())
            .enumerate()
            .filter(|(_, (_, ty))| matches!(ty, TokenType::Control | TokenType::UserDefined))
            .map(|(i, (t, _))| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let (byte_enc, byte_dec) = build_byte_maps();

        let kvu32 = |key: &str| g.kv(key).and_then(Value::as_u64).map(|v| v as u32);
        let kvbool = |key: &str| g.kv(key).and_then(Value::as_bool);

        Ok(Tokenizer {
            bos: kvu32("tokenizer.ggml.bos_token_id"),
            eos: kvu32("tokenizer.ggml.eos_token_id"),
            add_bos: kvbool("tokenizer.ggml.add_bos_token").unwrap_or(false),
            add_eos: kvbool("tokenizer.ggml.add_eos_token").unwrap_or(false),
            tokens,
            types,
            id_of,
            ranks,
            specials,
            pre,
            byte_enc,
            byte_dec,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.types.get(id as usize).copied()
    }

    /// Tokenize. `parse_special` = treat control/user-defined token strings in
    /// `text` as single tokens (template rendering path uses true).
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if parse_special && !self.specials.is_empty() {
            let mut rest = text;
            while !rest.is_empty() {
                // earliest match wins; specials are longest-first so equal
                // positions prefer the longer token.
                let mut best: Option<(usize, usize, u32)> = None; // (pos, len, id)
                for (s, id) in &self.specials {
                    if let Some(pos) = rest.find(s.as_str()) {
                        let better = match best {
                            None => true,
                            Some((bp, bl, _)) => pos < bp || (pos == bp && s.len() > bl),
                        };
                        if better {
                            best = Some((pos, s.len(), *id));
                        }
                    }
                }
                match best {
                    Some((pos, len, id)) => {
                        self.encode_plain(&rest[..pos], &mut out);
                        out.push(id);
                        rest = &rest[pos + len..];
                    }
                    None => {
                        self.encode_plain(rest, &mut out);
                        break;
                    }
                }
            }
        } else {
            self.encode_plain(text, &mut out);
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut start = 0;
        while let Ok(Some(m)) = self.pre.find_from_pos(text, start) {
            // find_from_pos with a well-formed pretokenizer always advances;
            // guard anyway so a zero-width match can't loop forever.
            if m.end() == start {
                break;
            }
            // the pretokenizer should partition the text; if it ever leaves a
            // gap, that text is still tokenized (never silently dropped)
            if m.start() > start {
                self.bpe_piece(&text[start..m.start()], out);
            }
            self.bpe_piece(&text[m.start()..m.end()], out);
            start = m.end();
        }
        if start < text.len() {
            self.bpe_piece(&text[start..], out);
        }
    }

    fn bpe_piece(&self, piece: &str, out: &mut Vec<u32>) {
        // byte-encode: every UTF-8 byte becomes one symbol char
        let mut syms: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_enc[b as usize].to_string())
            .collect();
        if syms.is_empty() {
            return;
        }

        // merge loop: lowest rank first, leftmost on ties
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                let key = format!("{}\0{}", syms[i], syms[i + 1]);
                if let Some(&rank) = self.ranks.get(&key) {
                    if best.map_or(true, |(br, _)| rank < br) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let right = syms.remove(i + 1);
            syms[i].push_str(&right);
        }

        for s in &syms {
            match self.id_of.get(s) {
                Some(&id) => out.push(id),
                None => {
                    // byte-level BPE has all single bytes in vocab; a miss can
                    // only happen on corrupt vocab. Fall back per byte, skip
                    // what still misses (mirrors llama.cpp's no-unk behavior).
                    for ch in s.chars() {
                        if let Some(&id) = self.id_of.get(ch.to_string().as_str()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Detokenize. Control tokens render only when `render_special`.
    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(text) = self.tokens.get(id as usize) else {
                continue;
            };
            match self.types[id as usize] {
                TokenType::Normal | TokenType::Byte => {
                    for ch in text.chars() {
                        match self.byte_dec.get(&ch) {
                            Some(&b) => bytes.push(b),
                            None => bytes.extend_from_slice(ch.to_string().as_bytes()),
                        }
                    }
                }
                TokenType::Control => {
                    if render_special {
                        bytes.extend_from_slice(text.as_bytes());
                    }
                }
                TokenType::UserDefined => bytes.extend_from_slice(text.as_bytes()),
                TokenType::Unknown | TokenType::Unused => {}
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_maps_roundtrip() {
        let (enc, dec) = build_byte_maps();
        for b in 0..=255u8 {
            assert_eq!(dec[&enc[b as usize]], b);
        }
        assert_eq!(enc[b' ' as usize], '\u{120}'); // space -> Ġ
        assert_eq!(enc[b'\n' as usize], '\u{10A}'); // \n -> Ċ
        assert_eq!(enc[b'A' as usize], 'A');
    }

    #[test]
    fn pretokenizer_splits() {
        let re = Regex::new(PRE_QWEN35).unwrap();
        let text = "Hello, world! it's 42 spaces   end\n";
        let mut pieces = Vec::new();
        let mut start = 0;
        while let Ok(Some(m)) = re.find_from_pos(text, start) {
            pieces.push(&text[m.start()..m.end()]);
            start = m.end();
        }
        // digits split singly; "   end" splits as trailing-ws lookahead ("  ")
        // then " end"; the final \n rides the newline alternative.
        assert_eq!(
            pieces,
            ["Hello", ",", " world", "!", " it", "'s", " ", "4", "2", " spaces", "  ", " end", "\n"]
        );
    }
}
