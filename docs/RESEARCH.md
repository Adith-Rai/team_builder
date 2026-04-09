# Pokemon AI Research — What Professionals Have Done

Last updated: 2026-04-08 (Session 33 deep architecture comparison added)

This document summarizes the state of the art in Pokemon battle AI, what approaches worked, and how they compare to our project. Use this to inform architecture and training decisions.

---

## 0. SESSION 33 UPDATE — Deep architecture comparison (CANONICAL)

This section was added after a fresh round of source-code research on Metamon, VGC-Bench, and ps-ppo to settle architectural questions raised by the Session 33 plateau. **Read this first** — older sections below may have outdated assumptions (esp. about VGC-Bench's frame stacking and Metamon's specific dims).

### 0.1 Concrete architecture configs (from actual repo source)

Verified from `metamon/rl/configs/models/{small,medium,large}_agent.gin` and `vgc_bench/src/policy.py` and `ps-ppo/ppo_core.py`.

| Dim | **Ours v8/v9** | **VGC-Bench BCFP** | **Metamon Small (15M)** | **Metamon Medium (50M)** | **Metamon Large (200M)** | **ps-ppo (>1900 Elo)** |
|---|---|---|---|---|---|---|
| Per-step encoder d_model | **384** | **256** | 100 | 100 | 160 | unknown (likely 256-384) |
| Per-step layers | 4 | 3 | 3 | 3 | 5 | 4 |
| Per-step heads | 4 | 4 | 5 | 5 | 8 | **8** |
| Per-step ff_dim | **1536 (4× d_model)** | 256 (1× d_model!) | unknown | unknown | unknown | 2× d_model |
| Tokens per timestep | 16 | 13 (1 CLS + 12 mons) | ~10 (4 scratch + 6 num) | ~10 | ~10 | 15 (2 decision + 1 field + 12 mons) |
| **Temporal/sequence model** | **2L 4H 384d** | **NONE (frame stack optional)** | **3L 8H 512d** | **6L 8H 768d** | **9L 20H 1280d** | **NONE (stateless)** |
| Temporal context | 200 turns | n frames (opt) | 200 | 200 | 128 | n/a |
| Critic heads | 1 (distributional 51-bin) | 1 scalar | 4 (NCritics, popart) | 4 | 4 | 1 (distributional 51-bin) |
| Total params | **13.38M** | <10M est | 15M | 50M | 200M | unknown (~10-15M est) |
| Format | Gen 9 OU singles | VGC Doubles | Gens 1-4 OU | Gens 1-4 OU | Gens 1-4 OU | Gen 9 Random Battles |
| Compute (states) | ~5-6M | **5,013,504** | hundreds of M | hundreds of M | hundreds of M | **>250M** |
| Pool strategy | Filtered uniform (sp≥260, depth ~610) | **Uniform fictitious play (BCFP)** | "recent checkpoints" (recency-biased) | same | same | **No pool — single live policy** |
| Result | smart_avg ~52%, **Elo 1032** (extended ladder, SH-anchored) | **1768 Elo** | 41-58% GXE | higher | top 10% | **>1900 Elo** |

### 0.2 Stateless vs stateful — the actual situation

**Two of three published references are stateless** (or have stateless cores):

- **ps-ppo: fully stateless.** Confirmed via `inference.py`: `WeightStore` only stores current version, "refresh_snapshots_from_disk... unnecessary because the latest weights are synced continuously via RAM." No checkpoint pool, no temporal model, no opponent history. Temporal info encoded in **transition features** (who moved first, SE/immune/crit flags).
- **VGC-Bench: stateless core.** Confirmed via `vgc_bench/src/policy.py`: 3-layer transformer encoder with bidirectional self-attention, **no causal mask, no temporal axis by default**. Frame stacking + second time-axis encoder is **optional**, not part of the base BCFP setup that achieved 1768 Elo.
- **Metamon: heavily temporal.** Causal Transformer with d_model 512–1280, 3–9 layers across size variants, 200-turn context. "Relies entirely on memory to infer the opponent's team."

**The format correlation matters:**
- Random Battles (ps-ppo): no team preview, procedural sets, less long-arc strategy → stateless works
- VGC Doubles (VGC-Bench): combinatorial action space, short games (~10-25 turns avg) → stateless core sufficient
- OU singles (Metamon, us): team preview, hazards/status accumulation, win conditions across turns 30-60 → temporal helps

**Verdict for our format:** Metamon is the closest format match and uses heavy temporal. So temporal is justified for OU. **But our temporal is much smaller than Metamon's at every size variant**, and we have an unusually large per-step encoder for the temporal model we pair it with. We may have the capacity allocated in the wrong place.

### 0.3 The shape mismatch (the most important new finding)

Compare per-step encoder vs temporal sizing across references:

| Reference | Per-step d_model | Temporal d_model | Ratio (temp/per-step) |
|---|---|---|---|
| **Ours** | **384** | **384** | **1.00** |
| Metamon Small | 100 | 512 | 5.12 |
| Metamon Medium | 100 | 768 | 7.68 |
| Metamon Large | 160 | 1280 | 8.00 |
| VGC-Bench | 256 | n/a (no temporal) | n/a |
| ps-ppo | ~256-384 | n/a (no temporal) | n/a |

**Metamon's philosophy: lean per-step encoder, heavy sequence model.** The temporal transformer does most of the work. Per-step encoding is just enough to produce a meaningful timestep summary; the temporal model integrates these across the battle.

**Our philosophy (de facto, not by design): heavy per-step encoder, lean sequence model.** We put 384d/4L/4H into spatial and only 384d/2L/4H into temporal. The temporal module gets the smallest budget despite OU being the format that needs it most.

**This is a real architectural concern that's distinct from "model is too small overall."** Even if we kept the same total parameter count, redistributing capacity from spatial → temporal might be the right move. Untested.

### 0.4 Pool strategy — the empirical evidence

**VGC-Bench BCFP is the only published head-to-head pool comparison at our compute scale.** Their result:

| Pool method | Description | Result |
|---|---|---|
| **BCFP (Fictitious Play)** | BC pretrain → uniform sample over **all** past checkpoints | **WINS: 1768 Elo @ 1-team** |
| BCDO (Double Oracle / Nash) | BC pretrain → Nash equilibrium weighting from empirical payoff matrix | Second |
| BCSP (Self-Play) | BC pretrain → train against latest only | Third |

So **uniform-over-all-history beats both Nash-weighted and latest-only** in the closest-comparable regime to ours.

**ps-ppo uses pool=1** (latest only) and gets >1900 Elo, but at 50× our compute and on a shallower format. Their method doesn't transfer downward.

**Implication:** Our current setup (filtered uniform, sp≥260, ~610 deep) is structurally **the same as BCFP** with a quality filter. **The right empirical question isn't "is the pool too deep" but "is the filter throwing away signal."** The user's intuition that "uniform deep pool > recency-only" is supported by the BCFP result.

**Caveat (important historical context):** Our pool was NOT 610-deep for most of training history. Earlier runs capped at 100 or even smaller. The deep pool is only really used in the most recent ~500 iters. So we don't have a clean experiment showing deep-pool training across the full plateau period — the plateau may have come from a regime where pool was much smaller.

### 0.5 Compute scale — the comparison that matters

| Reference | States | Result |
|---|---|---|
| ps-ppo | ~250M | >1900 Elo (Random Battles) |
| **VGC-Bench BCFP** | **5,013,504** | **1768 Elo (VGC Doubles, internal scale)** |
| **Ours** | **~5-6M total** | **Elo 1032** (Session 33 extended ladder, SH=1000 anchor) |
| Metamon (online phase) | hundreds of M+ | Top 10% (OU) |

We're at almost exactly VGC-Bench's compute scale.

**⚠ SESSION 35 CRITICAL CORRECTION: The "700 Elo gap" is an artifact of incompatible scales.**

The VGC-Bench and our Elo ladders use completely different anchoring:

| Player | VGC-Bench Scale | Our Scale |
|---|---|---|
| Random | 1127 | 444 |
| SH | 1621 | 1000 |
| Best agent | 1768 (BCFP) | 1032 (snapshot_1784) |
| **Agent above SH** | **+147** | **+32** |

**Apples-to-apples: BCFP is +147 above SH; we are +32 above SH. The real gap is ~115 Elo,
not 700.** Still meaningful (~66% expected win rate for BCFP over us), but fundamentally
different from the "massive architectural chasm" framing that drove Sessions 33-34 planning.

Additionally, OU singles is harder than VGC doubles for AI: 30-60 turns vs 8-12, more hidden
info, hazard/status accumulation, longer credit assignment horizon. A stateless model suffices
for VGC; OU requires temporal modeling. Format-adjusted, the gap may be even smaller.

**Revised conclusion (Session 35):** The plateau is real but the gap is closable with targeted
fixes (hyperparameters, augmentation) rather than requiring fundamental architectural overhaul.
**1200+ iters of training between snapshot_0589 (Elo 1015) and snapshot_1784 (Elo 1032)
produced ~17 Elo of net change** — more training time at the current setup is still uneconomic.
But the lever is likely **hyperparameter fixes (lambda, entropy)** + **data augmentation**
before expensive architectural changes or BC scaling.

**The original "700 Elo gap → need bigger BC base" logic chain is broken.** At ~115 Elo gap,
cheaper experiments should be tried first. See §0.8 revised order of operations.

### 0.6 Eval methodology in published work

| Reference | Primary metric | Secondary |
|---|---|---|
| ps-ppo | Showdown ladder Elo | win % vs SimpleHeuristics |
| VGC-Bench | **Internal Elo round-robin** (BayesElo / pairwise) | win rate vs heuristic baselines |
| Metamon | **GXE (Glicko-2-derived rating)** + ladder rank | win rate vs heuristic baselines |

**All published references use Elo or rating-based primary metrics.** Heuristic-bot win rate is consistently the secondary check. We're the only one using bot-suite win rate as the primary signal — and Session 29 already documented that smart_avg is a poor predictor of H2H strength.

### 0.7 What this all means — concrete implications

Things research **supports** (don't change):
1. **Temporal model is justified for OU.** Metamon (closest format match) uses heavy temporal.
2. **Uniform-over-history pool is justified.** BCFP wins the pool A/B at our scale.
3. **Distributional value head is justified.** Both ps-ppo and Metamon (two-hot) use it.
4. **Entity tokenization is justified.** All references use it.
5. **BC pretrain → PPO fine-tune is justified.** Metamon, VGC-Bench, ps-ppo all do this.

Things research **questions about our setup** (Session 33 findings + Session 35 audit):
1. **Capacity allocation is inverted vs Metamon.** We have heavy per-step + light temporal. Metamon has light per-step + heavy temporal. Our temporal: 2L/4H/384d. Even Metamon Small's temporal: 3L/8H/512d. **Note (S35):** Entity tokenization (the actual breakthrough) is preserved at any spatial dim — shrinking spatial doesn't remove entity attention. Safer test: spatial 256d/3L + temporal 512d/3L.
2. ~~**Per-step ff_dim is doubled (768 = 2× d_model)**~~ **CORRECTED (S35):** actual code uses `ff_mult=4`, so ff_dim = 1536 = 4× d_model. This is the standard transformer ratio (Vaswani 2017). VGC-Bench's 1× is the outlier. **No change needed.**
3. **Single critic** vs Metamon's 4-critic ensemble with popart. Variance reduction benefit unmeasured.
4. **Attention head count: 4 vs ps-ppo's 8.** Free A/B test (param-neutral if d_head halves).
5. **Pool filter `sp≥260`** vs BCFP's no-filter. Consider recency-weighted sampling (70% recent / 30% old) per OpenAI Five's 80/20 split.
6. **⚠ GAE lambda = 0.75 is a major outlier (S35 finding).** Every published system uses 0.95 (VGC-Bench, ps-ppo, OpenAI Five, ProcGen). At 0.75, advantage estimates are heavily myopic — the model can't properly credit early-game moves for terminal reward in 30-60 turn battles. **Highest-priority fix.**
7. **Entropy coef = 0.04 is 4× standard (S35 finding).** VGC-Bench uses 0.001, ps-ppo 0.01, OpenAI Five 0.01. **However**, our history shows entropy collapsed to 0.51 at ent=0.02 during early training (Session 28-29, iters 159-199). Reduce cautiously to 0.02 first, not 0.01.

Things research **can't tell us** (updated S35):
1. ~~Whether our actual Elo is competitive with VGC-Bench's 1768.~~ **RESOLVED (S35):** Elo scales are incompatible. Apples-to-apples gap is ~115 Elo (§0.5), not 700.
2. **Whether lambda=0.95 will close the remaining gap.** Must test empirically.
3. **Whether slot permutation augmentation helps at our scale.** Theoretically sound, untested.

### 0.8 Order of operations (revised Session 35 — hyperparameter-first)

**⚠ SESSION 35 REVISION:** The Session 33 plan was built on a false "700 Elo gap." The real
gap is ~115 Elo (§0.5). This changes the optimal experiment order: **cheap hyperparameter
fixes first, expensive architectural changes only if cheap fixes don't close the gap.**

**New plan (canonical: `docs/NEXT_SESSION.md`):**

```
[Step a]  ✅ DONE — Elo measured at 1032 (extended ladder, 38 players, 703 matches)
[Step b]  ✅ DONE — Code refactor (Session 34)
   │
   ▼
[Exp 1]  Hyperparameter fix: --lam 0.95 --ent-coef 0.02    [CLI flags only, ~15 hrs]
   │      Run 200 iters from snapshot_1784, measure Elo vs baseline
   │      Decision: if +50 Elo → hyperparams were the bottleneck, continue
   │      if <+30 Elo → proceed to Exp 2
   │
   ▼
[Exp 2]  Slot permutation augmentation                       [~2 hrs impl + 200 iters]
   │      Randomly shuffle team/move slot order in build_turn_batch()
   │
   ▼
[Exp 3]  Recency-weighted pool (70% last 200, 30% older)    [Small code change + 200 iters]
   │
   ▼
[Exp 4]  Elo ladder measurement after Exp 1-3
   │      Decision: if total gain ≥+80 Elo → gap closed, proceed to multi-gen (Step c)
   │      if total gain <+50 Elo → capacity reallocation needed (Exp 5)
   │
   ▼
[Exp 5]  Capacity reallocation: spatial 256d/3L → temporal 512d/3L  [Requires BC retrain]
   │      Only if Exp 1-3 insufficient
   │
   ▼
[Step c]  Multi-gen prep + BC scaling (original c1-c6, now AFTER hyperparameter experiments)
   │       Still the long-term plan, but no longer the FIRST experiment
   │
   ▼
[Step d]  Cloud burst (after all local levers exhausted)
```

The original "30M BC scaling first" plan is **deprioritized** — it was premised on a 700 Elo
gap that doesn't exist. At ~115 Elo, hyperparameter fixes + augmentation may suffice.

---

## 1. Metamon (UT Austin, April 2025) — Top 10% on Ladder

**Paper:** "Human-Level Competitive Pokemon via Scalable Offline Reinforcement Learning with Transformers"
**Source:** https://arxiv.org/html/2504.04395v1
**Code:** https://github.com/UT-Austin-RPL/metamon
**Result:** Top 10% of active players, rank #31 in Gen1OU at peak

### Architecture
- **Causal Transformer** (not LSTM), 15M / 50M / 200M parameter variants
- Smallest useful model: 4.7M RNN ("Minikazam"), best public model: 142M ("Kakuna")
- **Shared backbone** with separate actor and critic output heads
- Support for multiple discount factors trained in parallel
- Built on **AMAGO** framework (off-policy RL on long sequences)
- Turn encoder processes (observation, previous_action, previous_reward) per timestep
- Transformer uses "summary tokens to attend over the multi-modal sequence"

### State Representation
- **Semi-readable text** (87 words) + **48 numerical features** per timestep
- Custom Pokemon vocabulary with `<unknown>` token for rare cases
- Text order standardized for consistent action semantics
- Only opponent's active Pokemon visible; full opponent team inferred from memory
- Previous rewards and actions included for temporal reasoning
- **Key**: text tokens carry semantic meaning — "Earthquake" and "Charizard" are tokens the model can learn associations for, unlike flat numerical vectors

### Training Pipeline (the key progression)
1. **Imitation Learning (IL):** RNN baselines (500K-4M params) trained on BC
2. **Transformer IL:** 15-200M parameter models with pure BC → 41-58% GXE
3. **Offline RL:** Actor-critic with advantage filtering (not IQL/CQL/DT — custom)
   - Actor loss: `L = -w(h,a)*log_pi(a|h) - lambda * E[Q(h,a)]`
   - w(h,a) variants: IL (w=1, λ=0), Exp (w=exp(βA), λ=0), Binary (w=[A>0], λ=0), Binary+MaxQ (w=[A>0], λ>0)
   - **Binary+MaxQ won**: filter to positive advantage actions AND maximize Q-values
   - Binary threshold: **advantage > 0** (not median — simpler than we implemented)
   - **Two-hot value classification** for critic (converts value regression to classification bins, references MuZero). Critical for stability.
4. **Self-Play Data:** 2-5M synthetic trajectories per stage from diverse agent interactions
5. **Fine-tuning on self-play:** Final "SynRL-V2" model → top 10%

### Hyperparameters
- γ = 0.999 (very long horizon, multiple horizons trained in parallel)
- Reward: win/loss dominated + light shaping for damage dealt and health recovered
- Self-play teams: "Variety Team Set" — ~1K procedurally generated teams per tier (intentionally unrealistic for diversity)

### Dataset
- 475K human replays reconstructed from Showdown spectator logs (2014-2024)
- Each battle yields 2 perspectives → 950K sequences, 38M timesteps
- Extended to 4M+ trajectories after replay reconstruction
- Then 18M+ self-play trajectories on top
- Innovation: "replay reconstruction" — converting spectator logs to agent POV using team distribution statistics to infer hidden information

### Key Insights
1. **Long-context memory is critical** — full battle history, not just current state
2. **Self-play data was THE breakthrough** — 18M synthetic trajectories from diverse agent matchups. This is what took them from 58% GXE to top 10%
3. **Multi-format training** (16 rulesets simultaneously) improved robustness
4. **Out-of-distribution diversity helps** — "intentionally unrealistic" synthetic teams prevented overfitting to human team distributions
5. **Critic accuracy matters** — two-hot value classification dramatically improved offline RL
6. **Binary advantage filtering > exponential weighting** for offline RL
7. Models occasionally accumulate errors in exceptionally long battles

### Performance Progression
| Stage | Performance |
|-------|------------|
| RNN IL (4M params) | Baseline |
| Large Transformer IL | 41-58% GXE |
| Offline RL (SynRL-V1) | 64-80% GXE |
| Self-play fine-tune (SynRL-V2) | **Top 10% of players** |

---

## 2. MIT Thesis — Jett Wang (2024) — Rank 8, 1693 Elo

**Paper:** "Winning at Pokemon Random Battles Using Reinforcement Learning"
**Source:** https://dspace.mit.edu/handle/1721.1/153888
**Result:** Rank 8 (1693 Elo) on gen4randombattles ladder — best non-human agent for this format

### Method
- **PPO + self-play** to train actor-critic neural network
- Then use the neural network to **guide Monte Carlo Tree Search (MCTS)**
- RL alone plateaus, but RL + search breaks through
- The NN provides a value function and action prior for tree search

### Key Insight
**Search + learned value function is more powerful than pure policy.** The neural net learns "what's generally good" via self-play, then MCTS explores specific situations more deeply at inference time. This is the AlphaGo/AlphaZero paradigm applied to Pokemon.

---

## 3. PokeChamp (ICML 2025) — 1300-1500 Elo, Top 10-30%

**Paper:** "PokeChamp: an Expert-level Minimax Language Agent"
**Source:** https://arxiv.org/html/2503.04094v1
**Code:** https://github.com/sethkarten/pokechamp
**Result:** 1300-1500 Elo on ladder (top 10-30%), 84% vs best heuristic bot

### Architecture
LLMs replace three classical minimax components:
1. **Action sampling:** LLM proposes viable moves (reduces branching factor)
2. **Opponent modeling:** Statistical estimation of hidden stats + LLM prediction of opponent actions
3. **Value estimation:** LLM evaluates position at leaf nodes of search tree

### Additional Details
- One-step lookahead using official damage calc formula + historical data
- Turns-to-KO heuristic (info available to human players on ladder)
- Severe time constraint: 150 seconds total per player, 15 seconds per turn
- Uses 3M+ game dataset for statistical opponent inference

### Performance
- 84% vs Abyssal heuristic bot (Elo 1268)
- 76% vs PokeLLMon (GPT-4o)
- 64% vs PokeLLMon using only Llama 3.1 8B
- ~1/3 of online games timed out (search is slow)

### Key Insights
1. **Constrain search space with knowledge** — LLM narrows actions to strategically viable ones
2. **Hybrid symbolic-neural** — exact damage calc (symbolic) + LLM position eval (neural)
3. **Weaknesses:** Can't handle stall strategies (limited lookahead), excessive switching
4. No training needed — uses pre-trained LLM knowledge. But expensive per-game inference.

---

## 4. PokeLLMon (2024) — 49% Win Rate on Ladder

**Source:** https://arxiv.org/html/2402.01118v2
**Result:** 49% win rate on Gen8 Random Battles ladder, 56% in invited battles

### Method
- LLM (GPT-4) conditioned on observation history, actions, turn results
- Retrieval-augmented generation from Pokemon knowledge database
- No training — pure in-context learning

### Key Insight
LLMs have significant Pokemon knowledge from internet pretraining, but pure prompting hits a ceiling without search or explicit reasoning.

---

## 5. VGC-Bench (2025) — Doubles Benchmark

**Source:** https://arxiv.org/html/2506.10326v3
**Code:** https://github.com/cameronangliss/vgc-bench
**Result:** Single-team agents can beat VGC professionals

### Architecture
- **3-layer transformer encoder** for spatial (processes 12 Pokemon per battle)
- Second transformer with causal mask for temporal (frame stacking over history)
- **Separate actor and critic** (no parameter sharing, same architecture)
- Embeddings for moves, items, abilities
- Observation: `12×(global + side + per-pokemon features)`, frame-stacked: `n×12×(...)`
- Invalid actions masked to -∞ with softmax

### Training
- **PPO** with γ=1.0, λ=0.95, clip=0.2, ent=0.001, lr=**1e-5**, vf_coef=0.5, grad_clip=0.5
- Batch size 64, 10 PPO epochs per update, steps_per_update = 24×128
- **5,013,504 total timesteps** per agent
- 8× A40 GPUs + 2× Intel Xeon CPUs
- **Terminal reward only** (±1 win/loss)

### Population Methods
- **Self-play (SP)**: agent trains against itself
- **Fictitious play (FP)**: maintains checkpoint pool, trains against uniform distribution of past policies
- **Double oracle (DO)**: Nash equilibrium distribution over empirical payoff matrix
- **BC + self-play (BCSP)**: BC pretrain on 700K+ human replays, then fine-tune with self-play
- BCSP was recommended baseline

### Key Findings
- Single-team agents beat VGC professionals (2020 Dallas Regional Champion)
- Generalization across teams is the hard problem — best single-team method degrades with larger team pools
- BC pretraining + self-play is a validated pipeline

---

## 6. ps-ppo / Nebraskinator (2025) — >1900 Elo, Gen 9 Random Battles

**Code:** https://github.com/Nebraskinator/ps-ppo
**Result:** >1900 Elo on Gen9 Random Battle ladder, >85% vs SimpleHeuristicsPlayer. Highest documented pure neural policy (no tree search).

### Architecture — PokeTransformer
- **15-token transformer**: 2 decision tokens (Actor/Critic) + 1 field token + 12 Pokemon tokens
- 4 layers, 8 heads, GELU, pre-layer norm
- **Poke-Mask**: state tokens can't see decision tokens and vice versa. Actor can't see Critic. Forces pure state representation.
- **Distributional value head**: 51-bin categorical (support [-1.6, 1.6]) with two-hot encoding. NOT scalar regression.
- Entity embeddings: species, moves, items, abilities as learned embeddings
- "Numerical banks": learned embeddings for continuous values (HP, stats, BP) instead of raw floats

### State Representation (>3000 features)
Per Pokemon (12 total): HP%, base stats (via embedding banks), stat boosts (one-hot 13 bins per stat), types (multi-hot), status, volatiles, level, weight, height, tera type, 4 moves (ID + BP + accuracy + PP + type + category + priority)

**Transition features (critical)**: Previous move IDs for both players, who moved first, super effective flag, resisted flag, immune flag, critical hit flag. "Allows the model to 'see' the immediate past, which is vital for high-level decision making."

**NOT included**: type effectiveness calculations, damage estimates. Model learns these from entity embeddings + experience.

### Training
- **Phase 1**: BC on SimpleHeuristicsPlayer demonstrations until perfect imitation
- **Phase 2**: PPO self-play via Ray distributed framework
- **250M states**, 2 days on RTX 3090
- Reward: +/-1 win/loss, +/-0.1 per faint

### Key Design Decisions
1. **Transformer over MLP**: "MLP struggled to internalize tactical depth" (~1100 Elo). Transformers preserve entity relationships.
2. **Entity tokenization**: Each Pokemon is a separate token. Transformer directly attends "my move" to "opponent Pokemon" — no MLP bottleneck.
3. **Binned/embedded continuous values**: Raw floats cause "gradient instability." Learned embedding banks work better.
4. **Poke-Mask**: Separation of state representation from policy/value prevents gradient interference.
5. **Distributional value**: Captures uncertainty in outcomes, gives richer gradient signal than scalar regression.

---

## 7. Self-Play RL for Pokemon (IEEE CoG 2019)

**Source:** https://ieeexplore.ieee.org/iel7/8844551/8847948/08848014.pdf
**Code:** https://github.com/yuzeh/metagrok

### Method
- Self-play policy optimization
- Early work establishing that self-play can learn Pokemon strategies without human data

---

## Patterns Across All Successful Approaches

### What works
| Pattern | Used by | Our v7 status |
|---------|---------|------------|
| Transformers (long context) | Metamon, ps-ppo, PokeChamp | DONE (20.78M, 6L/4H/512d) |
| Self-play (population-based) | Metamon, MIT, VGC-Bench, ps-ppo | DONE (pool + HoF + historical) |
| BC -> self-play pipeline | Metamon, VGC-Bench, ps-ppo | DONE (BC → PPO, skipped IQL) |
| 15-200M params | Metamon, ps-ppo | DONE (20.78M) |
| **Entity tokenization** | **ps-ppo** | **MISSING — flat vector + MLP bottleneck** |
| **Distributional value head** | **Metamon, ps-ppo** | **MISSING — scalar regression** |
| **Previous action + outcome features** | **Metamon, ps-ppo** | **MISSING** |
| **Structured attention (Poke-Mask)** | **ps-ppo** | **MISSING** |
| Search at inference time | MIT, PokeChamp | Not yet |
| Multi-format training | Metamon | Single format |

### What doesn't work
| Pattern | Evidence |
|---------|----------|
| Pure BC alone | All: BC ceiling ~20-40% GXE |
| IQL / offline RL | Us: 3 runs, all failed. Q-network can't learn action-level values from offline data |
| Flat MLP encoding | ps-ppo: MLP baseline ~1100 Elo. Transformer needed for entity relationships |
| Scalar value regression | Metamon: two-hot classification "critical for stability" |
| Fixed-bot-only PPO | Us: plateaued at 27%. Self-play required. |
| Adaptive opponent weights | Us: causes beat-A-forget-B oscillation |
| Recent-only snapshot pool | Us: causes ~100 iter strategy cycling (aggressive → stalling → pivoting) |

### Feature Comparison Across All Approaches

| Feature | Us | Metamon | ps-ppo (1900 Elo) | VGC-Bench |
|---------|-----|---------|-------------------|-----------|
| **Architecture** | Flat → MLP → transformer (timesteps) | Text → transformer (timesteps) | **Entity tokens → transformer (entities)** | Entity tokens → transformer (spatial + temporal) |
| **Obs → model flow** | 1480d → MLP → 512d → transformer | 87 text + 48 num → transformer | >3000d → 15 tokens → transformer | 12×(g+s+p) → transformer |
| **Entity relationships** | Crushed by MLP bottleneck | Implicit in text sequence | **Direct attention between entities** | Direct attention between entities |
| **Value head** | Scalar regression | Two-hot classification | **51-bin distributional** | Scalar |
| **Previous action** | Opponent only | **Own + opponent** | **Own + opponent + who moved first** | Unknown |
| **Previous outcome** | None | **Previous reward** | **SE/resisted/immune/crit flags** | Unknown |
| **Type effectiveness** | Not computed | Not computed | Not computed | Unknown |
| **Attention masking** | None | None | **Poke-Mask (state↔decision separation)** | Causal (temporal) |
| **Data scale** | ~16M steps | 22M+ trajectories | **250M states** | 5M steps |

---

## Implications for v8 (Updated After v7 Findings)

### The Real Gaps (discovered through v7 experiments + research comparison)

The v7 self-play run proved that the training PIPELINE works (BC → PPO self-play with population diversity). But the model plateaus at 20-23% vs smart bots because of ARCHITECTURAL limitations, not training limitations.

**1. Entity Tokenization (HIGHEST PRIORITY)**
Our model crushes 1480 features through an MLP into 512 dims before the transformer sees it. This destroys entity relationships. ps-ppo's architecture processes each Pokemon as a separate token — the transformer directly attends "my Earthquake" to "opponent Charizard" and learns type relationships through attention weights. ps-ppo explicitly found that MLP baseline reached only ~1100 Elo while their entity-token transformer reached >1900.

**2. Previous Action + Outcome Features**
Both Metamon and ps-ppo include what happened last turn in the observation. ps-ppo encodes: which move was used, who went first, super effective flag, resisted flag, immune flag, critical hit. This enables cause-effect learning: "I used Earthquake → it was immune → opponent is Flying type → don't use it again." Our model must reconstruct this implicitly from HP changes across turns, which is much harder.

**3. Distributional Value Head**
Both Metamon (two-hot) and ps-ppo (51-bin categorical) use classification bins instead of scalar regression. This captures outcome uncertainty and provides richer gradient signal. Our scalar value head was likely a factor in IQL's failure (weak advantage signal) and PPO's instability.

**4. Structured Attention Masking**
ps-ppo's Poke-Mask separates state representation from policy/value decisions. This prevents the policy from warping the state representation — the battle state is encoded "objectively" and the policy/value heads independently read from it.

### What we DON'T need (corrected from earlier analysis)
- **Type effectiveness features**: NONE of the successful approaches inject these. ps-ppo explicitly does NOT compute type effectiveness. The model learns it from entity embeddings + experience IF the architecture allows entity-level attention.
- **Damage estimates**: Same — not used by any successful approach.
- **More data alone**: ps-ppo used 250M states but the architecture is what matters. Our 16M states with the right architecture should be sufficient to start.

### v8 Roadmap (prioritized)
1. **Entity tokenization** — restructure features.py + policy_heads.py. Each Pokemon becomes a token with its own features. Transformer processes entities, not flat vectors. This is the ps-ppo architecture.
2. **Previous action + outcome features** — add to features.py: our last action (move/switch ID), who moved first, SE/resisted/immune/crit flags from last turn. ~20 new features.
3. **Distributional value head** — replace scalar value with N-bin categorical. Two-hot encoding for targets. Moderate policy_heads.py change.
4. **Poke-Mask attention** — separate actor/critic/state attention. Prevents gradient interference.
5. **Numerical embedding banks** — ps-ppo embeds continuous values (HP, stats, BP) through learned banks instead of raw floats. "Causes gradient instability" with raw floats.
