//! Choosing how deep to draft, per round.
//!
//! Lives here rather than in the CLI because the server needs it more: a fixed depth is
//! a bet that the text is predictable, and on unpredictable text speculation is a net
//! loss — measured at 30.5 tok/s against 32.5 for not speculating at all, on prose with
//! 0.21 acceptance. Code sits at the other end (0.67 acceptance, 67.8 tok/s at depth 3),
//! and a server does not get to know which it is being asked for.

/// Chooses the draft-chain length for each round.
///
/// Throughput at chain length K is `E(K) / T(K)`:
///   `E(K) = 1 + sum_{j=1..K} prod_{i<=j} p_i` — tokens the round commits
///   `T(K)`                                    — what the round costs
///
/// `p_j` is the chance the j-th draft survives given its predecessors did, which is
/// exactly what a round already reports: drafts `1..n_keep` were accepted and, if any
/// remained, draft `n_keep+1` was not.
///
/// `T(K)` is measured per depth rather than fitted globally. A least-squares fit over
/// `(K, time)` looked tidier but is ill-conditioned in the case that actually matters —
/// once the picker settles, nearly every sample is at one depth — and it produced
/// nonsense slopes there.
///
/// The search is a hill climb over adjacent depths, not a sweep. A sweep pays for the
/// deep end on every prompt, and the deep end is exactly where a round is most
/// expensive; probing a neighbour occasionally costs one round in twelve and still
/// finds the peak, because throughput in K is single-peaked.
pub struct DepthPicker {
    max: usize,
    adaptive: bool,
    fixed: usize,
    hits: Vec<f64>,
    trials: Vec<f64>,
    /// per-depth round time, seconds; None until that depth has run
    time: Vec<Option<f64>>,
    /// steady-state timing samples per depth, so a depth timed once — and therefore
    /// timed together with building its graph — is not mistaken for solid evidence
    samples: Vec<usize>,
    current: usize,
    rounds: usize,
    chosen: Vec<usize>,
}

impl DepthPicker {
    /// How fast acceptance evidence is forgotten. A ~69-round half-life looks far too
    /// long to track a generation moving from prose into code into a table, and 0.95
    /// (~13 rounds) was tried for exactly that reason — it measured *worse* on three of
    /// four prompts. The extra responsiveness buys noise, not tracking: what actually
    /// mis-ranked the depths was the cost model, not the acceptance model.
    const ACCEPT_DECAY: f64 = 0.99;
    /// Weight for the round-time average, applied only after a depth's first couple of
    /// samples are discarded: those rounds also paid to build that shape's graph.
    /// A minimum was tried instead and is subtly wrong — the more rounds a depth gets,
    /// the more chances it has to record an unusually fast one, so the depth already in
    /// use always looks cheapest and the picker talks itself into staying.
    const TIME_EMA: f64 = 0.2;
    const WARM_SAMPLES: usize = 2;
    /// Switch on a small modelled win. A wider margin was safer when a switch created a
    /// new graph shape; with reductions trimmed to the depth they are moving to, moving
    /// between depths is nearly free, and the model understates the deeper option
    /// anyway — it cannot know how well position K+1 is accepted until it has run there.
    const SWITCH_MARGIN: f64 = 1.01;
    /// Optimism bonus at zero evidence, in units of acceptance. Large enough that an
    /// unmeasured depth is worth one visit, small enough to be swamped after a handful
    /// of rounds there.
    const EXPLORE: f64 = 0.3;

    pub fn new(adaptive: bool, fixed: usize, max: usize) -> Self {
        let max = max.max(1);
        let mut me = Self {
            max,
            adaptive,
            fixed,
            hits: vec![0.0; max + 2],
            trials: vec![0.0; max + 2],
            time: vec![None; max + 2],
            samples: vec![0; max + 2],
            current: 1,
            rounds: 0,
            chosen: vec![0; max + 2],
        };
        // Open at whatever depth the prior alone considers best, rather than a guessed
        // constant: with no evidence the model already encodes "acceptance starts high
        // and decays", and its optimum under that belief is the right opening bid. It
        // also means the opening move follows the cost model instead of contradicting
        // it if the hardware constants ever change.
        let prof = me.profile();
        me.current = (1..=me.max)
            .max_by(|a, b| {
                let sa = me.tokens_per_round(*a, &prof) / me.cost(*a);
                let sb = me.tokens_per_round(*b, &prof) / me.cost(*b);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(1);
        me
    }

    /// Acceptance per chain position, each inheriting the one before it as its prior.
    ///
    /// A flat prior (an untried position reads as a coin flip) is what stopped this
    /// climbing: on text where every draft is being accepted, the depth one step deeper
    /// still scored as a gamble, so the picker never went to look. Inheriting is the
    /// better assumption in both directions — acceptance decays smoothly along a chain,
    /// so if position 3 survives 95% of the time, position 4 is far likelier to be near
    /// 0.95 than near 0.5, and on hard text the same rule keeps the estimate low.
    fn profile(&self) -> Vec<f64> {
        self.profile_with(0.0)
    }

    /// Acceptance per chain position, each inheriting the one before it as its prior,
    /// plus an optimism bonus that decays as evidence arrives.
    ///
    /// The prior does NOT decay along the chain, which is worth spelling out because it
    /// is the opposite of the obvious assumption. Measured conditional acceptance:
    ///
    /// | prompt | p1 | p2 | p3 |
    /// |---|---|---|---|
    /// | prose | 0.88 | 0.72 | 0.63 |
    /// | code | 0.94 | 0.55 | **1.00** |
    /// | arithmetic | 1.00 | 0.91 | ~1.00 |
    ///
    /// Acceptance *rises* along the chain on two of the three. It is a survivorship
    /// effect: position 3 is only ever reached when positions 1 and 2 were accepted,
    /// and that selects for a locally predictable stretch of text, where the next draft
    /// is likely to be right too. A decaying prior therefore mis-prices exactly the
    /// case where depth pays, which is how the picker came to sit at depth 2 on code
    /// while depth 3 was 5% faster.
    ///
    /// `explore` is a UCB bonus, scaled by how little evidence a position has. It is
    /// what makes the picker willing to run at depth K+1 at all: without it, position
    /// K+1 can never be priced, because it is only observed by going there.
    fn profile_with(&self, explore: f64) -> Vec<f64> {
        const PRIOR: f64 = 2.0;
        let mut out = vec![1.0; self.max + 2];
        let mut prev = 0.85; // opening belief for position 1
        for j in 1..=self.max {
            let n = self.trials[j];
            let mean = (self.hits[j] + PRIOR * prev) / (n + PRIOR);
            out[j] = (mean + explore / (n + PRIOR).sqrt()).min(1.0);
            prev = mean; // inherit the estimate, never the optimism
        }
        out
    }

    fn p(&self, j: usize) -> f64 {
        self.profile()[j.min(self.max)]
    }

    /// Round cost in seconds.
    ///
    /// Anchored on the one depth with real evidence and extrapolated with a per-step
    /// constant, rather than measured independently per depth. A depth the picker has
    /// only visited to probe is timed two or three times at most, and those rounds also
    /// paid to build that shape's graph — which made the cheapest depth look like the
    /// most expensive one and pinned the picker where it started. The per-round and
    /// per-step costs are hardware, not text: measured at 25.8 ms + 6.05 ms/step on the
    /// 27B across two 3090s, with the same slope on prompts of very different
    /// difficulty.
    /// Fallback per-chain-step cost, used until two depths have been timed.
    ///
    /// Forcing a visit to the shallower depth to get that second timing was tried and
    /// is a clear loss: five calibration rounds cost 8% on a 46-round generation,
    /// because a shallower round also produces one fewer draft for the round after it.
    /// The depths the picker moves between on its own supply the second timing whenever
    /// it moves at all, and a generation where it never moves is one where it is
    /// confident enough that the slope does not change the answer.
    const PER_STEP: f64 = 0.00605;

    fn anchor(&self) -> Option<(usize, f64)> {
        (1..=self.max)
            .filter(|k| self.samples[*k] > Self::WARM_SAMPLES + 2)
            .filter_map(|k| self.time[k].map(|t| (k, t)))
            .max_by_key(|(k, _)| self.samples[*k])
    }

    /// What one more chain step costs, measured from two timed depths when both have
    /// been visited, and otherwise taken as a constant.
    ///
    /// The constant is trustworthy, which is worth recording because it was doubted:
    /// measured directly across generations of 128, 256 and 512 tokens, ms/round was
    /// 32.0/37.6/44.1, 31.9/37.8/44.1 and 31.9/37.9/44.9 at depths 1/2/3 — a per-step
    /// cost of 5.6-7.0 ms with no trend in context. An earlier reading of ~12.7 ms at
    /// long context was an artefact of applying one run's acceptance profile to a
    /// different run's throughput, and is withdrawn; the fit is kept because it
    /// self-corrects if the hardware or model changes, not because context moves it.
    fn per_step(&self) -> f64 {
        let mut pts: Vec<(usize, f64)> = (1..=self.max)
            .filter(|k| self.samples[*k] > Self::WARM_SAMPLES + 2)
            .filter_map(|k| self.time[k].map(|t| (k, t)))
            .collect();
        if pts.len() >= 2 {
            pts.sort_by_key(|(k, _)| *k);
            let (k0, t0) = pts[0];
            let (k1, t1) = *pts.last().unwrap();
            let s = (t1 - t0) / (k1 - k0) as f64;
            // reject a slope that cannot be a chain step, so one odd round cannot
            // invert the ranking
            if s > 0.001 && s < 0.05 {
                return s;
            }
        }
        Self::PER_STEP
    }

    fn cost(&self, k: usize) -> f64 {
        match self.anchor() {
            Some((a, t)) => (t + self.per_step() * (k as f64 - a as f64)).max(0.005),
            None => 0.0258 + Self::PER_STEP * k as f64,
        }
    }

    fn tokens_per_round(&self, k: usize, prof: &[f64]) -> f64 {
        let mut acc = 1.0;
        let mut run = 1.0;
        for j in 1..=k {
            run *= prof[j];
            acc += run;
        }
        acc
    }

    pub fn choose(&mut self) -> usize {
        if !self.adaptive {
            self.chosen[self.fixed.min(self.max)] += 1;
            return self.fixed;
        }
        self.rounds += 1;
        if self.rounds > 3 {
            let prof = self.profile_with(Self::EXPLORE);
            let score = |k: usize| self.tokens_per_round(k, &prof) / self.cost(k);
            let mut best = self.current;
            let mut best_score = score(self.current);
            for k in [self.current.saturating_sub(1).max(1), (self.current + 1).min(self.max)] {
                if score(k) > best_score * Self::SWITCH_MARGIN {
                    best = k;
                    best_score = score(k);
                }
            }
            self.current = best;
        }
        self.chosen[self.current] += 1;
        self.current
    }

    /// `carried` is how many drafts the batch actually verified, which is what the
    /// acceptance trial is over; `chain` is how many the round produced. They differ
    /// only on the round a depth switch takes effect, and such a round costs something
    /// between the two depths — so it is left out of the timing rather than polluting
    /// it.
    pub fn observe(&mut self, carried: usize, chain: usize, n_keep: usize, secs: f64) {
        for v in self.hits.iter_mut().chain(self.trials.iter_mut()) {
            *v *= Self::ACCEPT_DECAY;
        }
        // positions 1..n_keep were reached and accepted; n_keep+1 was reached and
        // rejected. Anything past that was never reached, so it is not evidence.
        for j in 1..=n_keep.min(carried) {
            self.trials[j] += 1.0;
            self.hits[j] += 1.0;
        }
        if n_keep < carried {
            self.trials[n_keep + 1] += 1.0;
        }
        if carried != chain || chain > self.max {
            return;
        }
        self.samples[chain] += 1;
        if self.samples[chain] <= Self::WARM_SAMPLES {
            return;
        }
        self.time[chain] = Some(match self.time[chain] {
            // a round far above the running estimate is a rebuild, not the steady cost
            Some(t) if secs > 1.5 * t => t,
            Some(t) => t + Self::TIME_EMA * (secs - t),
            None => secs,
        });
    }

    pub fn report(&self) -> String {
        if !self.adaptive {
            return String::new();
        }
        let hist: Vec<String> = (1..=self.max)
            .filter(|k| self.chosen[*k] > 0)
            .map(|k| format!("d{k}x{}", self.chosen[k]))
            .collect();
        let acc: Vec<String> = (1..=self.max)
            .filter(|k| self.trials[*k] > 0.5)
            .map(|k| format!("p{k}={:.2}", self.p(k)))
            .collect();
        let ms: Vec<String> = (1..=self.max)
            .map(|k| format!("t{k}={:.0}", self.cost(k) * 1000.0))
            .chain(std::iter::once(format!("step={:.1}", self.per_step() * 1000.0)))
            .collect();
        format!(
            ", adaptive [{}] {} | {}ms",
            hist.join(" "),
            acc.join(" "),
            ms.join(" ")
        )
    }
}

