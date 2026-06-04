# Replay Rehearsal — AWR vs Off-Policy PPO

Technical reference for the replay-rehearsal-during-PPO experiment. AWR is the
first-try recipe; off-policy PPO is the "next-level" path if AWR validates the
direction.

Companion to: `docs/PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md` (strategic context),
`memory/project_plateau_hypothesis_and_experiments.md` (memory pointer),
Task #125 (concrete implementation).

---

## Why this matters

Our hypothesis (S68): BC v10 was trained on 1500+ Elo replays, exposing model
to elite play patterns. PPO with 100% procedural teams never gives those
patterns reward signal → they fade. Replay rehearsal continues to expose model
to elite states/actions DURING PPO so the patterns get reinforced (or
refined-and-rejected) via training signal.

Two flavors of replay rehearsal differ in HOW the replay signal updates the
policy:

---

## AWR (Advantage-Weighted Regression)

**Loss:**

```
L_AWR = - E_(s,a)~replay [ exp(A(s,a) / β) * log π_θ(a | s) ]
```

with the binary-filter variant:

```
L_AWR_binary = - E_(s,a)~replay [ 1[A(s,a) > 0] * log π_θ(a | s) ]
```

**What it does mechanistically:** for each replay (s, a):
1. Compute reward `R` from the human battle outcome using OUR reward formula
   (terminal + ko_coef + hp_coef + immune_penalty)
2. Forward our value head to get `V_θ(s)`
3. Compute `Advantage = R - V_θ(s)`
4. Compute loss as weighted BC: high-advantage actions get strong gradient
   pulling π_θ(a|s) up; low-advantage actions get tiny (or zero) gradient

**Crucial property — no behavior policy needed.** AWR doesn't ask "what was
the probability the human took this action?" It only asks "did this action
work out well?" That's why AWR works on data where the behavior policy is
unknown (e.g., anonymous human replays).

**Conceptual framing**: AWR is "BC that increases likelihood of high-reward
actions." Model trains AS IF these actions were what it should be doing,
weighted by how well they worked. User's intuition "trick the model into
thinking it fought like an elite and got those results" maps directly to AWR.

**Hyperparameters worth tuning:**
- `β` (temperature): smaller = sharper filtering (more like binary). β=1 is
  metamon's default.
- `clip_weights_high`: max advantage weight to prevent extreme single-sample
  domination. Metamon uses 100.
- Mix-in weight (0.05-0.1) into total PPO loss: controls how much replay
  signal vs on-policy PPO signal drives updates.

---

## Off-Policy PPO

**Loss:**

```
L_offpolicy_PPO = - E_(s,a)~replay [ min( ratio * A, clip(ratio, 1-ε, 1+ε) * A ) ]
```

where `ratio = π_θ(a|s) / π_behavior(a|s)`.

**What it does mechanistically:** proper policy gradient with importance
sampling correction:
1. Compute Advantage from replay as in AWR
2. Compute ratio = our current policy probability / human's behavior probability
3. PPO clip + min provides natural stabilization (limits how far the policy
   can drift from where the data came from)

**Crucial property — requires behavior policy estimate.** `π_behavior(a|s)` is
the probability the original actor (human) took action a in state s. For real
replays, this is UNKNOWN.

**Workarounds for unknown `π_behavior`:**
- **Approximate with BC v10's distribution**: since v10 was trained to mimic
  these same humans, its π is a reasonable proxy. Add forward pass on v10 per
  replay sample. Bias-vs-variance tradeoff. Simplest.
- **Uniform over legal actions**: extremely biased but very simple. The ratio
  becomes π_θ(a|s) × N_legal_actions. Variance can be high.
- **V-trace style truncation**: bound the ratio to [c_min, c_max] (e.g.,
  [0.01, 100]). Reduces variance, introduces some bias. Standard in IMPALA.

---

## Concrete example walkthrough

Replay state s: opponent's Dragapult is in, has 30% HP. Human played
`Earthquake` (assume hitting Dragapult's switch target), eventually won the
battle.

Our current policy outputs:
- π_θ(Earthquake | s) = 0.30
- π_θ(Stealth Rock | s) = 0.40
- π_θ(other moves | s) = 0.30

Assume reward function gives R = +1 (won the battle). Our V_θ(s) = +0.5
(model expected to win slightly).

**Advantage = R - V_θ(s) = +0.5**

### AWR update

- Weight = `exp(0.5 / 1.0) = 1.65`
- Loss = `-1.65 * log(0.30) = -1.65 * (-1.20) = +1.98` (positive loss to minimize)
- Gradient: backprop through `log π_θ(Earthquake | s)`, weighted by 1.65
- **Net effect**: pulls `π_θ(Earthquake | s)` UP toward higher probability.
  No signal about Stealth Rock or other actions (not in this sample).
- No behavior policy needed.

### Off-policy PPO update

Approximate `π_behavior(Earthquake | s) ≈ 0.40` (BC v10's probability).

- `ratio = 0.30 / 0.40 = 0.75`
- `clipped_ratio = clip(0.75, 1-0.2, 1+0.2) = 0.80`
- Loss = `-min(0.75 * 0.5, 0.80 * 0.5) = -min(0.375, 0.40) = -0.375` (we want to
  MINIMIZE the negative, so gradient pushes ratio up → π_θ(Earthquake|s) up)
- **Net effect**: similar direction as AWR (pull Earthquake probability up)
  but smaller magnitude (scaled by ratio 0.75 instead of weight 1.65) and
  bounded by clip
- **IS correction matters**: if our policy gave π_θ(Earthquake|s) = 0.10 (much
  less than human's 0.40), ratio = 0.25 → small update despite positive
  advantage. This is the "trust region" property — don't update too aggressively
  on out-of-distribution actions.

---

## When each shines

| Scenario | Choose |
|---|---|
| Unknown behavior policy (human replays) | **AWR** |
| Implementation simplicity matters | **AWR** |
| Cheapest first-try experiment | **AWR** |
| Validating "replays help" hypothesis | **AWR** |
| You have lots of off-policy data | Off-policy PPO |
| Want proper RL guarantees | Off-policy PPO |
| Want to use existing PPO infrastructure | Off-policy PPO |
| Want IS correction for distribution shift | Off-policy PPO |
| Already have behavior policy estimate | Off-policy PPO |

---

## Comparison table (theoretical properties)

| Aspect | AWR | Off-policy PPO |
|---|---|---|
| Update style | Weighted BC | Policy gradient |
| Importance sampling | No | Yes (via ratio) |
| Behavior policy needed | No | Yes (or approximate) |
| Asymptotic bias | Some (BC bias toward high-A actions even at convergence) | Lower (IS correction → consistent if IS accurate) |
| Variance | Low (BC-style; only the weight varies) | Higher (IS ratios can spike) |
| Compute overhead | 1 extra V forward / sample | V + behavior-policy forward / sample |
| Hyperparameter sensitivity | Medium (β, weight) | Higher (clip ε, IS truncation bounds, weight) |
| Theoretical guarantees | Weaker (BC-bias toward high-A actions) | Stronger (proper RL update if IS correct) |
| Implementation complexity | Low (~1 day) | Medium (2-3 days) |

---

## Why AWR is the right first try

1. **No behavior policy estimation needed.** Real replays don't have human
   action probabilities; estimating them is its own subproject.
2. **Proven on this exact domain.** Metamon's exp_rl.gin and binary_rl.gin
   (used by LargeRL, MediumRL_Aug, SyntheticRLV2, Minikazam) are AWR variants.
   They produce competitive models on parsed-replays-v4 data we'd use for
   rehearsal. We know AWR works on this data shape.
3. **Cheapest test of the hypothesis.** If AWR with reasonable hyperparameters
   doesn't help, the hypothesis is probably weak and off-policy PPO's added
   complexity won't save it.
4. **Lower variance.** AWR's BC-style updates are stable; we're already
   navigating reward-hack territory (bc_kl > 0.20 in fishbowl_v2_resume) and
   high-variance off-policy updates could destabilize.
5. **User's framing maps directly.** "Trick the model into thinking it fought
   like an elite and got those results" IS what AWR does.

---

## When to escalate to off-policy PPO

Off-policy PPO becomes worth the implementation cost when:

1. **AWR validates the direction but plateaus.** If AWR-PPO mixed training shows
   improvement vs baseline but stops short of where we want to be, the AWR-bias
   may be limiting us. Off-policy PPO's proper RL gradient could push further.
2. **We want to leverage multi-step credit assignment.** AWR uses one-step
   advantage; off-policy PPO can use GAE/n-step returns from replays.
3. **We have a good behavior-policy estimate.** If we end up training a BC v11
   on the same replays first, that gives us π_behavior for free.
4. **We want training stability under heavy off-policy load.** If we ever want
   >30% off-policy data, AWR may become unstable; off-policy PPO with clipping
   is more robust.

---

## Implementation gotchas if/when we move to off-policy PPO

- **Behavior-policy estimation**: BC v10 (current) wasn't trained for the
  observation space currently used by the production model. Need to either
  retrain a BC variant matching the current obs space, OR adapt v10's outputs
  to current obs space (forward pass with care). If mismatched, IS ratios will
  be systematically biased.
- **Truncation bounds**: extreme ratios (rare actions) blow up updates. Standard
  is V-trace-style truncation: cap rho at ~1.0, c at ~1.0 (per IMPALA / R2D2
  conventions). Tune per environment.
- **Memory cost**: storing replay batches with (s, a, V_θ(s), π_behavior(a|s))
  per sample is heavier than just (s, a) for AWR.
- **Cold-start instability**: at the start of PPO, V_θ may be inaccurate →
  bad advantage estimates → bad updates. Mitigate via warmup (only AWR for
  first N iters, then enable off-policy PPO).

---

## Common ground (regardless of variant)

Both variants share:

- **Replay data source**: `jakegrigsby/metamon-raw-replays` filtered gen9ou
  rating ≥ 1500 (the same dataset we used for BC v10).
- **State encoding**: must match our current observation space exactly. If we
  ever change observation space (e.g., S58 encoding-gaps fix), need to re-encode
  replays.
- **Reward computation on replays**: need a function that takes a replay
  outcome (winner, final HPs, KOs, etc.) and emits our standard shaped reward
  (terminal + ko_coef + hp_coef + immune_penalty). May require small adapter
  code.
- **Value head forward on replay states**: O(1) per sample, batched with
  current PPO updates.

---

## Implementation steps for AWR (Task #125)

1. **Replay loader**: extend or reuse BC v10 data loader. Yield batches of
   (encoded_state, action, terminal_outcome).
2. **Reward function on replay outcomes**: small adapter wrapping our existing
   reward shaping. Input: replay battle metadata. Output: scalar R per step.
3. **V_θ forward on replay states**: standard model forward pass; cache for
   the iter.
4. **AWR loss term**: `-mean(exp(A / β) * log_softmax(logits).gather(a))`. Mix
   into PPO total loss with weight 0.05.
5. **CLI flags**: `--replay-rehearsal-path PATH` (replay dataset), `--replay-mix-weight 0.05`,
   `--awr-beta 1.0`, `--awr-binary` (use binary filter instead of exp).
6. **Validation run**: fishbowl-scale from snap_0139, with vs without replay
   rehearsal. Compare smart_avg, MM-vs-our_model WRs, ladder placement after.

---

## See also

- `docs/PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md` — strategic context, related experiments
- `memory/project_plateau_hypothesis_and_experiments.md` — memory index
- Task #125 — concrete implementation tracking
- Metamon `exp_rl.gin` / `binary_rl.gin` — proven AWR variants in our domain
- AWR paper: Peng et al. 2019, "Advantage-Weighted Regression"
- IMPALA / V-trace paper: Espeholt et al. 2018 (for IS truncation patterns
  relevant to off-policy PPO)
