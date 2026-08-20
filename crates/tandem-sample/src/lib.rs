//! Token sampling.
//!
//! Everything in tandem up to now has decoded greedily, because the accuracy gate is
//! temp-0 parity against llama.cpp. Serving needs the rest, and the semantics here are
//! ported deliberately from llama.cpp's `llama-sampler.cpp` at the pinned build so that
//! the same request parameters mean the same thing in both engines.
//!
//! What that does and does not buy: the *filters* (top-k, top-p, min-p, penalties) are
//! reproduced exactly, so the candidate set for a given set of logits is identical. The
//! final draw is not, and cannot be — llama.cpp draws with `std::mt19937` through
//! `std::discrete_distribution`, whose stream is implementation-defined. So a seeded run
//! matches itself, never llama.cpp token for token. Token-exact parity remains a
//! temperature-0 claim, and at temperature 0 this samples argmax, which is what the
//! existing gates already check.
//!
//! Ordering follows llama.cpp's default chain: penalties -> top-k -> top-p -> min-p ->
//! temperature -> draw. It matters: the filters run on untempered probabilities.

use std::collections::VecDeque;

/// One candidate token during sampling.
#[derive(Clone, Copy, Debug)]
struct Cand {
    id: u32,
    logit: f32,
    p: f32,
}

#[derive(Clone, Debug)]
pub struct SamplerParams {
    /// <= 0 selects the argmax, matching llama.cpp's `temp_impl`.
    pub temp: f32,
    /// 0 disables.
    pub top_k: usize,
    /// >= 1.0 disables.
    pub top_p: f32,
    /// <= 0.0 disables.
    pub min_p: f32,
    /// Never let a filter cut the candidate set below this.
    pub min_keep: usize,
    /// 1.0 disables. Applied to tokens seen in the last `penalty_last_n`.
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalty_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            min_keep: 1,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerParams {
    /// True when the chain can only ever return the argmax, so the caller can keep using
    /// the in-graph argmax and skip reading a vocabulary of logits back over PCIe.
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0 && self.penalty_repeat == 1.0 && self.penalty_freq == 0.0
            && self.penalty_present == 0.0
    }
}

/// xoshiro256++, seeded through splitmix64.
///
/// Self-contained on purpose: the engine has no random dependency, and a sampler whose
/// stream is defined here is one whose runs can be reproduced from the seed alone.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits is the whole mantissa; taking the high bits avoids the weak low bits
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct Sampler {
    params: SamplerParams,
    rng: Rng,
    /// recent tokens, most recent last, capped at `penalty_last_n`
    history: VecDeque<u32>,
    cand: Vec<Cand>,
}

impl Sampler {
    pub fn new(params: SamplerParams) -> Self {
        let rng = Rng::new(params.seed);
        Self { params, rng, history: VecDeque::new(), cand: Vec::new() }
    }

    pub fn params(&self) -> &SamplerParams {
        &self.params
    }

    /// Record a token as generated, for the penalty window.
    pub fn accept(&mut self, token: u32) {
        if self.params.penalty_last_n == 0 {
            return;
        }
        self.history.push_back(token);
        while self.history.len() > self.params.penalty_last_n {
            self.history.pop_front();
        }
    }

    /// Seed the penalty window from an existing context (a prompt, say).
    pub fn accept_all(&mut self, tokens: &[u32]) {
        for t in tokens {
            self.accept(*t);
        }
    }

    /// Pick a token from a full vocabulary of logits.
    ///
    /// `logits` is not modified; penalties and temperature are applied to the candidate
    /// set this builds.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let d = self.distribution(logits);
        let r = self.rng.next_f32();
        d.draw(r)
    }

    /// The distribution this sampler would draw from, as a value.
    ///
    /// Speculative decoding needs it rather than just a draw: verifying a drafted token
    /// means asking how likely the *target* was to produce it, and on rejection drawing
    /// from what is left. Returning the distribution keeps that arithmetic in one place
    /// and keeps it exactly the distribution `sample` would have used.
    pub fn distribution(&mut self, logits: &[f32]) -> Dist {
        self.build(logits);
        Dist { cand: self.cand.iter().map(|c| (c.id, c.p)).collect() }
    }

    /// Draw with an explicit uniform, so a caller sequencing several draws controls the
    /// order they consume randomness in.
    pub fn draw_from(&mut self, d: &Dist) -> u32 {
        let r = self.rng.next_f32();
        d.draw(r)
    }

    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }

    fn build(&mut self, logits: &[f32]) {
        assert!(!logits.is_empty(), "sampling needs a vocabulary");
        let p = self.params.clone();

        // Penalties first, on the full vocabulary, since they can promote a token into
        // the candidate set as easily as demote one out of it.
        let mut penalised: Vec<(u32, f32)> = Vec::new();
        if !(p.penalty_repeat == 1.0 && p.penalty_freq == 0.0 && p.penalty_present == 0.0) {
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for t in &self.history {
                *counts.entry(*t).or_insert(0) += 1;
            }
            for (tok, count) in counts {
                let i = tok as usize;
                if i >= logits.len() {
                    continue;
                }
                let mut l = logits[i];
                // Dividing alone would make an already-negative logit *more* likely,
                // so the sign decides the direction — same fix llama.cpp applies.
                if l <= 0.0 {
                    l *= p.penalty_repeat;
                } else {
                    l /= p.penalty_repeat;
                }
                l -= count as f32 * p.penalty_freq + p.penalty_present;
                penalised.push((tok, l));
            }
        }
        // A linear scan here would be O(vocab x window) — 16M comparisons per token at a
        // 64-token penalty window on this vocabulary.
        let penalised: std::collections::HashMap<u32, f32> = penalised.into_iter().collect();
        let logit_of = |i: usize, base: &[f32]| -> f32 {
            match penalised.get(&(i as u32)) {
                Some(l) => *l,
                None => base[i],
            }
        };

        // Narrow before sorting. The vocabulary here is 248,320 entries and a full sort
        // per token would cost more than the forward pass; every path below is either
        // O(n) or a partial sort over the survivors.
        self.cand.clear();
        if p.top_k > 0 && p.top_k < logits.len() {
            let mut all: Vec<Cand> = (0..logits.len())
                .map(|i| Cand { id: i as u32, logit: logit_of(i, logits), p: 0.0 })
                .collect();
            let k = p.top_k.max(p.min_keep).min(all.len());
            all.select_nth_unstable_by(k - 1, |a, b| cmp_desc(a, b));
            all.truncate(k);
            all.sort_unstable_by(cmp_desc);
            self.cand = all;
        } else if p.min_p > 0.0 {
            // min-p is a threshold on the logit, so it filters without any sort:
            // p_i >= min_p * p_max is exactly logit_i >= max_logit + ln(min_p).
            let mut max_l = f32::NEG_INFINITY;
            for i in 0..logits.len() {
                max_l = max_l.max(logit_of(i, logits));
            }
            let floor = max_l + p.min_p.ln();
            for i in 0..logits.len() {
                let l = logit_of(i, logits);
                if l >= floor {
                    self.cand.push(Cand { id: i as u32, logit: l, p: 0.0 });
                }
            }
            self.cand.sort_unstable_by(cmp_desc);
        } else if p.top_p < 1.0 {
            // top-p needs probabilities normalised over the whole vocabulary, but only
            // the head of the distribution can matter. Normalise in O(n), then take a
            // probe of the largest few and widen only if they do not already cover p —
            // sorting 248,320 candidates per token costs more than the forward pass.
            let mut all: Vec<Cand> = (0..logits.len())
                .map(|i| Cand { id: i as u32, logit: logit_of(i, logits), p: 0.0 })
                .collect();
            softmax(&mut all);
            let mut probe = 1024.min(all.len());
            loop {
                all.select_nth_unstable_by(probe - 1, cmp_desc);
                let covered: f32 = all[..probe].iter().map(|c| c.p).sum();
                if covered >= p.top_p || probe >= all.len() {
                    break;
                }
                probe = (probe * 8).min(all.len());
            }
            all.truncate(probe);
            all.sort_unstable_by(cmp_desc);
            self.cand = all;
        } else {
            // nothing narrows the set; temperature and the draw are both O(n)
            self.cand = (0..logits.len())
                .map(|i| Cand { id: i as u32, logit: logit_of(i, logits), p: 0.0 })
                .collect();
        }

        if p.temp <= 0.0 {
            // argmax: a point mass, which is what makes a greedy draft always accepted
            // when the same code verifies it
            let mut best = 0usize;
            for i in 1..self.cand.len() {
                if self.cand[i].logit > self.cand[best].logit {
                    best = i;
                }
            }
            let winner = self.cand[best];
            self.cand.clear();
            self.cand.push(Cand { p: 1.0, ..winner });
            return;
        }

        // top-p runs on untempered probabilities, as in llama.cpp's chain. The `p`
        // values are already normalised over the whole vocabulary by the narrowing
        // above; re-normalising over the probe would renormalise away the very tail
        // top-p exists to cut.
        if p.top_p < 1.0 && self.cand.len() > 1 {
            if p.top_k > 0 || p.min_p > 0.0 {
                softmax(&mut self.cand);
            }
            let mut cum = 0.0f32;
            let mut last = self.cand.len();
            for i in 0..self.cand.len() {
                cum += self.cand[i].p;
                // the crossing element is included
                if cum >= p.top_p && i + 1 >= p.min_keep {
                    last = i + 1;
                    break;
                }
            }
            self.cand.truncate(last);
        }
        // min-p again for the top-k path, where the threshold has not been applied yet
        if p.min_p > 0.0 && p.top_k > 0 && self.cand.len() > 1 {
            let floor = self.cand[0].logit + p.min_p.ln();
            let mut keep = 1;
            while keep < self.cand.len()
                && (self.cand[keep].logit >= floor || keep < p.min_keep)
            {
                keep += 1;
            }
            self.cand.truncate(keep);
        }

        for c in self.cand.iter_mut() {
            c.logit /= p.temp;
        }
        softmax(&mut self.cand);
    }
}

/// A normalised distribution over the tokens that survived filtering.
#[derive(Clone, Debug)]
pub struct Dist {
    cand: Vec<(u32, f32)>,
}

impl Dist {
    /// How likely this distribution was to produce `token`. Zero if filtering removed
    /// it, which is the case a speculative verifier must reject outright.
    pub fn prob_of(&self, token: u32) -> f32 {
        self.cand
            .iter()
            .find(|(id, _)| *id == token)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    pub fn draw(&self, r: f32) -> u32 {
        let mut cum = 0.0f32;
        for (id, p) in &self.cand {
            cum += *p;
            if r < cum {
                return *id;
            }
        }
        self.cand.last().expect("a distribution is never empty").0
    }

    /// Draw from this distribution with `token` removed and the rest renormalised —
    /// the residual a speculative round samples from when it rejects a draft.
    pub fn draw_excluding(&self, token: u32, r: f32) -> u32 {
        let total: f32 = self
            .cand
            .iter()
            .filter(|(id, _)| *id != token)
            .map(|(_, p)| *p)
            .sum();
        if total <= 0.0 {
            // the excluded token was all of the mass; nothing else can be drawn
            return self.draw(r);
        }
        let target = r * total;
        let mut cum = 0.0f32;
        for (id, p) in &self.cand {
            if *id == token {
                continue;
            }
            cum += *p;
            if target < cum {
                return *id;
            }
        }
        self.cand
            .iter()
            .rev()
            .find(|(id, _)| *id != token)
            .expect("checked non-empty above")
            .0
    }

    /// The largest probability in the distribution — how confident the model is at this
    /// position. Flat here means a draft is unlikely to survive verification.
    pub fn peak(&self) -> f32 {
        self.cand.iter().map(|(_, p)| *p).fold(0.0f32, f32::max)
    }

    pub fn len(&self) -> usize {
        self.cand.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cand.is_empty()
    }
}

fn cmp_desc(a: &Cand, b: &Cand) -> std::cmp::Ordering {
    // descending by logit; ties broken by id so the order is total and reproducible
    b.logit
        .partial_cmp(&a.logit)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.id.cmp(&b.id))
}

fn softmax(cand: &mut [Cand]) {
    let max_l = cand.iter().fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
    let mut sum = 0.0f32;
    for c in cand.iter_mut() {
        c.p = (c.logit - max_l).exp();
        sum += c.p;
    }
    for c in cand.iter_mut() {
        c.p /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(p: SamplerParams) -> Sampler {
        Sampler::new(p)
    }

    #[test]
    fn temp_zero_is_argmax() {
        let logits = [0.1, 5.0, -2.0, 4.9];
        for top_k in [0, 1, 2, 4] {
            for top_p in [1.0, 0.5] {
                let mut smp = s(SamplerParams { top_k, top_p, ..Default::default() });
                assert_eq!(smp.sample(&logits), 1, "top_k {top_k} top_p {top_p}");
            }
        }
    }

    #[test]
    fn top_k_one_is_argmax_at_any_temperature() {
        let logits = [1.0, 3.0, 2.0];
        let mut smp = s(SamplerParams { temp: 2.0, top_k: 1, ..Default::default() });
        for _ in 0..50 {
            assert_eq!(smp.sample(&logits), 1);
        }
    }

    #[test]
    fn top_p_includes_the_crossing_token() {
        // softmax([2,1,0]) = .665, .245, .090
        // top_p 0.7 must keep two: the running sum only reaches 0.7 at the second.
        let logits = [2.0, 1.0, 0.0];
        let mut smp = s(SamplerParams { temp: 1.0, top_p: 0.7, ..Default::default() });
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            seen.insert(smp.sample(&logits));
        }
        assert_eq!(seen, [0u32, 1].into_iter().collect());
    }

    #[test]
    fn min_p_thresholds_relative_to_the_peak() {
        // p = .665, .245, .090; min_p 0.5 keeps only tokens with p >= 0.5*0.665 = .332
        let logits = [2.0, 1.0, 0.0];
        let mut smp = s(SamplerParams { temp: 1.0, min_p: 0.5, ..Default::default() });
        for _ in 0..500 {
            assert_eq!(smp.sample(&logits), 0);
        }
        // 0.3 keeps two (.245 >= .199), not three (.090 < .199)
        let mut smp = s(SamplerParams { temp: 1.0, min_p: 0.3, ..Default::default() });
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4000 {
            seen.insert(smp.sample(&logits));
        }
        assert_eq!(seen, [0u32, 1].into_iter().collect());
    }

    #[test]
    fn repeat_penalty_pushes_a_repeated_token_down() {
        let logits = [5.0, 4.9];
        let mut smp = s(SamplerParams { penalty_repeat: 2.0, ..Default::default() });
        assert_eq!(smp.sample(&logits), 0);
        smp.accept(0); // now token 0 is penalised: 5.0 / 2 = 2.5 < 4.9
        assert_eq!(smp.sample(&logits), 1);
    }

    #[test]
    fn repeat_penalty_does_not_promote_negative_logits() {
        // dividing a negative logit by the penalty would raise it; the sign check
        // multiplies instead, which must push it further down
        let logits = [-1.0, -1.5];
        let mut smp = s(SamplerParams { penalty_repeat: 2.0, ..Default::default() });
        assert_eq!(smp.sample(&logits), 0);
        smp.accept(0); // -1.0 * 2 = -2.0, now below -1.5
        assert_eq!(smp.sample(&logits), 1);
    }

    #[test]
    fn a_seed_reproduces_a_run() {
        let logits = [1.0, 1.1, 0.9, 1.05];
        let run = |seed| {
            let mut smp = s(SamplerParams { temp: 1.0, seed, ..Default::default() });
            (0..64).map(|_| smp.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn the_draw_follows_the_distribution() {
        // logits [ln 1, ln 3] -> 25% / 75%
        let logits = [0.0f32, 3.0f32.ln()];
        let mut smp = s(SamplerParams { temp: 1.0, seed: 7, ..Default::default() });
        let n = 40_000;
        let ones = (0..n).filter(|_| smp.sample(&logits) == 1).count();
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.75).abs() < 0.01, "got {frac}, expected ~0.75");
    }

    #[test]
    fn temperature_flattens() {
        let logits = [0.0f32, 3.0f32.ln()];
        let hot = {
            let mut smp = s(SamplerParams { temp: 8.0, seed: 3, ..Default::default() });
            (0..20_000).filter(|_| smp.sample(&logits) == 1).count()
        };
        let cold = {
            let mut smp = s(SamplerParams { temp: 0.25, seed: 3, ..Default::default() });
            (0..20_000).filter(|_| smp.sample(&logits) == 1).count()
        };
        assert!(hot < cold, "hotter sampling must be closer to uniform: {hot} vs {cold}");
        assert!(cold > 19_000, "cold sampling should almost always take the peak");
    }

    #[test]
    fn a_greedy_distribution_is_a_point_mass() {
        // this is what makes a greedy draft always accepted by the same verifier
        let logits = [1.0, 4.0, 2.0];
        let mut smp = s(SamplerParams::default());
        let d = smp.distribution(&logits);
        assert_eq!(d.len(), 1);
        assert_eq!(d.prob_of(1), 1.0);
        assert_eq!(d.prob_of(0), 0.0);
    }

    #[test]
    fn a_filtered_token_has_no_probability() {
        let logits = [2.0, 1.0, 0.0];
        let mut smp = s(SamplerParams { temp: 1.0, top_k: 1, ..Default::default() });
        let d = smp.distribution(&logits);
        assert_eq!(d.prob_of(0), 1.0);
        // token 2 did not survive top-k, so a draft proposing it must be rejected
        assert_eq!(d.prob_of(2), 0.0);
    }

    #[test]
    fn draw_excluding_never_returns_the_excluded_token() {
        let logits = [2.0, 1.0, 0.0];
        let mut smp = s(SamplerParams { temp: 1.0, ..Default::default() });
        let d = smp.distribution(&logits);
        for i in 0..1000 {
            let r = i as f32 / 1000.0;
            assert_ne!(d.draw_excluding(0, r), 0);
        }
    }

    /// The property speculative decoding at temperature rests on.
    ///
    /// A draft proposes the argmax, so its own distribution is a point mass. Accepting
    /// it with probability `p(x0)` and otherwise drawing from the residual emits
    /// exactly `p` — which is what lets a sampled request keep speculating without
    /// changing what it would have produced.
    #[test]
    fn the_acceptance_rule_reproduces_the_target_distribution() {
        let logits = [1.0f32, 0.5, 0.0, -0.5];
        let mut smp = s(SamplerParams { temp: 1.0, seed: 11, ..Default::default() });
        let target = smp.distribution(&logits);
        let x0 = 0u32; // what a greedy draft head would propose

        let n = 200_000;
        let mut counts = [0u32; 4];
        for _ in 0..n {
            let accept_roll = smp.rng().next_f32();
            let tok = if accept_roll < target.prob_of(x0) {
                x0
            } else {
                let r = smp.rng().next_f32();
                target.draw_excluding(x0, r)
            };
            counts[tok as usize] += 1;
        }
        for t in 0..4u32 {
            let got = counts[t as usize] as f32 / n as f32;
            let want = target.prob_of(t);
            assert!(
                (got - want).abs() < 0.005,
                "token {t}: emitted {got:.4}, target {want:.4}"
            );
        }
    }

    #[test]
    fn is_greedy_reports_when_the_readback_can_be_skipped() {
        assert!(SamplerParams::default().is_greedy());
        assert!(!SamplerParams { temp: 0.7, ..Default::default() }.is_greedy());
        // a penalty changes which token wins even at temperature 0
        assert!(!SamplerParams { penalty_repeat: 1.1, ..Default::default() }.is_greedy());
    }
}
