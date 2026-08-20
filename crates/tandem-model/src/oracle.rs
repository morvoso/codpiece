//! CPU-side context oracle: draft tokens that cost the GPU nothing.
//!
//! # Why this exists on this machine
//!
//! Decode on 2×3090 is memory-bandwidth-bound: a verify pass reads the whole
//! weight set exactly once whether it checks 1 token or 12. So the cost of a
//! speculative round is essentially
//!
//! ```text
//!   round = one weight-read (fixed)  +  sum of draft costs
//! ```
//!
//! and throughput is `accepted_tokens / round`. That makes *draft cost* the
//! thing to attack, not draft quality: a draft that is free improves
//! throughput even at low acceptance, because it can only add accepted
//! tokens to a pass we were paying for anyway.
//!
//! The MTP head is a real transformer block — one attention layer plus a
//! 248k-row LM projection — so each draft it produces costs measurable GPU
//! time (~8 ms on the 27B, measured). Its first draft is worth that (0.90
//! acceptance); its third is not (0.67 and falling, at full price).
//!
//! Meanwhile this box has a 24-thread i9 and 62 GiB of RAM sitting idle
//! during every decode step. This module puts them to work: it predicts
//! continuations from the token stream itself, in microseconds, off the GPU's
//! critical path entirely.
//!
//! # The predictor
//!
//! A bounded-order backoff model over the *live* token stream (prompt plus
//! everything generated so far), stored as a hash map from an n-gram to the
//! tokens that followed it and how often.
//!
//! Longer contexts are tried first and we fall back on miss, so a match is
//! specific when it can be and still fires when it cannot. Prediction is
//! several hash lookups; update is a handful of inserts per committed token.
//!
//! This is deliberately not a neural drafter. It is exceptional at exactly
//! the things transformers spend tokens on and speculative decoders love —
//! verbatim quotation from the prompt, repeated identifiers in code, list and
//! table scaffolding, closing brackets and tags, boilerplate phrasing — and
//! useless at genuinely novel text, where it declines to predict rather than
//! guessing. Declining is the correct behavior: an unfilled draft slot costs
//! nothing, while a wrong one costs a rollback.
//!
//! # How it composes with MTP
//!
//! The two drafters are good at different things, so tandem uses both in one
//! round: MTP supplies the first draft (highest quality, worth its cost), and
//! the oracle extends the round for free. Extending a verify batch is close
//! to free on bandwidth-bound hardware, so the extra slots are close to pure
//! upside.

use std::collections::HashMap;

/// Longest n-gram context the oracle will key on. Beyond this the extra
/// specificity stops paying for the lookup and the memory.
const MAX_ORDER: usize = 8;
/// Shortest context worth keying on. Order 1 predicts from a single token,
/// which is mostly noise at this vocabulary size.
const MIN_ORDER: usize = 2;

/// Counts of tokens observed following one context.
#[derive(Default, Clone)]
struct Followers {
    /// (token, count), kept sorted by count descending; short by construction
    items: Vec<(u32, u32)>,
}

impl Followers {
    fn observe(&mut self, tok: u32) {
        if let Some(i) = self.items.iter().position(|(t, _)| *t == tok) {
            self.items[i].1 += 1;
            let mut i = i;
            while i > 0 && self.items[i].1 > self.items[i - 1].1 {
                self.items.swap(i, i - 1);
                i -= 1;
            }
        } else {
            self.items.push((tok, 1));
            // a context with many distinct continuations is not predictive;
            // keeping the head bounds both memory and lookup cost
            if self.items.len() > 8 {
                self.items.pop();
            }
        }
    }

    /// Best follower and its share of observations at this context.
    fn best(&self) -> Option<(u32, f32)> {
        let total: u32 = self.items.iter().map(|(_, c)| *c).sum();
        self.items
            .first()
            .map(|(t, c)| (*t, if total == 0 { 0.0 } else { *c as f32 / total as f32 }))
    }
}

/// Predicts continuations from the token stream, on the CPU, off the GPU's
/// critical path.
pub struct ContextOracle {
    /// key = FNV of (order, context tokens) -> followers
    table: HashMap<u64, Followers>,
    /// the live token stream (prompt + generated)
    stream: Vec<u32>,
    /// minimum share of observations required to emit a draft
    pub min_confidence: f32,
    pub hits: usize,
    pub proposals: usize,
    /// recent acceptance, exponentially weighted — the signal the gate tunes on
    recent: f32,
    /// how many drafts the recent estimate is based on
    seen: usize,
    /// when true, min_confidence self-tunes toward `target`
    pub adaptive: bool,
    /// acceptance the gate aims to hold
    pub target: f32,
}

fn hash_ctx(order: usize, ctx: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ (order as u64).wrapping_mul(0x9E3779B97F4A7C15);
    for t in ctx {
        h ^= *t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl ContextOracle {
    pub fn new(min_confidence: f32) -> ContextOracle {
        ContextOracle {
            table: HashMap::new(),
            stream: Vec::new(),
            min_confidence,
            hits: 0,
            proposals: 0,
            recent: 1.0,
            seen: 0,
            adaptive: true,
            target: 0.55,
        }
    }

    pub fn len(&self) -> usize {
        self.stream.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }

    /// Feed committed tokens. Only *committed* tokens go in — a rejected
    /// draft must never teach the oracle something the model did not say.
    pub fn extend(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.stream.push(t);
            let n = self.stream.len();
            for order in MIN_ORDER..=MAX_ORDER {
                if n < order + 1 {
                    break;
                }
                let ctx = &self.stream[n - order - 1..n - 1];
                self.table
                    .entry(hash_ctx(order, ctx))
                    .or_default()
                    .observe(t);
            }
        }
    }

    /// Predict the token following `tail`, longest context first.
    fn predict_one(&self, tail: &[u32]) -> Option<(u32, f32)> {
        for order in (MIN_ORDER..=MAX_ORDER).rev() {
            if tail.len() < order {
                continue;
            }
            let ctx = &tail[tail.len() - order..];
            if let Some(f) = self.table.get(&hash_ctx(order, ctx)) {
                if let Some((tok, conf)) = f.best() {
                    if conf >= self.min_confidence {
                        return Some((tok, conf));
                    }
                }
            }
        }
        None
    }

    /// Draft up to `n` tokens continuing `prefix` (committed stream plus any
    /// tokens already drafted this round). Stops at the first position it is
    /// not confident about — an empty slot is free, a wrong one is not.
    pub fn draft(&mut self, prefix: &[u32], n: usize) -> Vec<u32> {
        let mut tail: Vec<u32> = self
            .stream
            .iter()
            .rev()
            .take(MAX_ORDER)
            .rev()
            .copied()
            .collect();
        tail.extend_from_slice(prefix);

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.predict_one(&tail) {
                Some((tok, _)) => {
                    out.push(tok);
                    tail.push(tok);
                    if tail.len() > MAX_ORDER * 2 {
                        tail.drain(..MAX_ORDER);
                    }
                }
                None => break,
            }
        }
        self.proposals += out.len();
        out
    }

    /// Record how many of this round's oracle drafts the verify pass kept.
    ///
    /// This closes a control loop that costs nothing and runs on the CPU: the
    /// oracle's value is workload-dependent (measured 0.80 acceptance on
    /// prose that quotes itself, 0.26 while writing novel code), so rather
    /// than pick a threshold per workload, it watches what it is getting and
    /// tightens or loosens itself. Predicting less on unpredictable text is
    /// the correct response — those draft slots were free, and a wrong draft
    /// truncates the accepted prefix.
    pub fn record(&mut self, proposed_this_round: usize, accepted: usize) {
        self.hits += accepted;
        if proposed_this_round == 0 {
            return;
        }
        let rate = accepted as f32 / proposed_this_round as f32;
        // heavier weight early, then settle
        let alpha = if self.seen < 32 { 0.25 } else { 0.08 };
        self.recent = (1.0 - alpha) * self.recent + alpha * rate;
        self.seen += proposed_this_round;

        if !self.adaptive {
            return;
        }
        if self.recent < self.target {
            // demand more evidence before drafting
            self.min_confidence = (self.min_confidence + 0.05).min(0.95);
        } else if self.recent > self.target + 0.2 {
            // it is being too shy; free tokens are being left on the table
            self.min_confidence = (self.min_confidence - 0.03).max(0.20);
        }
    }

    pub fn confidence_gate(&self) -> f32 {
        self.min_confidence
    }

    pub fn acceptance(&self) -> f32 {
        if self.proposals == 0 {
            0.0
        } else {
            self.hits as f32 / self.proposals as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_repeated_structure() {
        let mut o = ContextOracle::new(0.5);
        // a repeating pattern, the shape of list and code scaffolding
        let pat: Vec<u32> = vec![10, 20, 30, 40];
        let mut stream = Vec::new();
        for _ in 0..6 {
            stream.extend_from_slice(&pat);
        }
        o.extend(&stream);
        // after "10, 20" the continuation is unambiguous
        let d = o.draft(&[10, 20], 2);
        assert_eq!(d, vec![30, 40], "should continue an established pattern");
    }

    #[test]
    fn gate_tightens_when_drafts_keep_missing() {
        let mut o = ContextOracle::new(0.4);
        let before = o.confidence_gate();
        for _ in 0..10 {
            o.record(2, 0); // proposed two, kept none
        }
        assert!(
            o.confidence_gate() > before,
            "a drafter that keeps missing must demand more evidence"
        );
    }

    #[test]
    fn gate_loosens_when_drafts_land() {
        let mut o = ContextOracle::new(0.8);
        let before = o.confidence_gate();
        for _ in 0..10 {
            o.record(2, 2); // everything kept
        }
        assert!(
            o.confidence_gate() < before,
            "a drafter that keeps landing should draft more"
        );
    }

    #[test]
    fn declines_when_unsure() {
        let mut o = ContextOracle::new(0.9);
        // same context followed by many different tokens: not predictive
        let mut stream = vec![1u32, 2];
        for t in 100..110u32 {
            stream.push(1);
            stream.push(2);
            stream.push(t);
        }
        o.extend(&stream);
        let d = o.draft(&[1, 2], 3);
        assert!(d.is_empty(), "an ambiguous context must not produce drafts");
    }

    #[test]
    fn quotes_the_prompt_back() {
        let mut o = ContextOracle::new(0.5);
        // the model quoting its input is the classic free-token case
        let prompt: Vec<u32> = (500..540).collect();
        o.extend(&prompt);
        // re-entering the quoted region mid-way
        let d = o.draft(&[520, 521, 522], 4);
        assert_eq!(d, vec![523, 524, 525, 526]);
    }

    #[test]
    fn only_committed_tokens_are_learned() {
        let mut o = ContextOracle::new(0.5);
        o.extend(&[1, 2, 3]);
        let before = o.len();
        let _ = o.draft(&[1, 2], 2);
        assert_eq!(o.len(), before, "drafting must not mutate the stream");
    }
}
