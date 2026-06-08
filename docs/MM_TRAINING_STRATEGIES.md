# Metamon MM Training Strategies — Reference

Created 2026-06-08 from `metamon_ref/` (shallow clone) — primarily
`metamon_ref/README.md` + `metamon_ref/metamon/rl/pretrained.py` +
`metamon_ref/metamon/rl/configs/training/*.gin`.

**Purpose**: understand HOW each MM we benchmark against was trained — so
we can interpret what our H2H WRs actually mean. **NOT for copying methodology** —
the user has been explicit on that. This is context-only.

> ⚠️ **GRAIN OF SALT**: Metamon documentation is incomplete and the repo
> is large. Some details below are inferred from gin configs + model
> naming conventions, not from explicit Metamon documentation.
> Re-verify against the live Metamon repo + their RLC 2025 paper
> ([arxiv:2504.04395](https://arxiv.org/abs/2504.04395)) before acting on
> any specific claim.

## TL;DR

The MMs we have in our training pool (LargeRL, MediumRL_Aug,
SyntheticRLV2, Minikazam) are **paper-era 2024 baselines** with a few
PokeAgent-Challenge additions. They are NOT Metamon's current
state-of-the-art (which is Kakuna at 142M params, Dec 2025). Even within
our set, the strongest opponent (Minikazam, 92% smart_avg) is the
SMALLEST model at 4.7M params with an RNN architecture — designed as an
"affordable starting point for finetuning."

All Metamon RL models share a common framework (**AMAGO**: offline RL
transformer), with the same base hyperparameters. The differences
between them are mostly:

1. Model size (50M / 200M / 4.7M)
2. Training data (1M human replays / +4M synthetic self-play /
   +PokeAgent challenge data)
3. Filter function (exp / binary / fake)
4. Architecture (multitask transformer / RNN)

## The 4 MMs in our training pool

### LargeRL (200M params)

- **Source**: paper era (RLC 2025), August 2024
- **Architecture**: AMAGO Perceiver multi-task transformer, "large" variant
- **Training method**: offline RL with **exponential filter (AWR-style)**
  - `agent.exp_filter.beta = 1.0` (soft advantage weighting)
  - `agent.exp_filter.clip_weights_high = 100`
  - offline_coeff=1.0, online_coeff=0.0 (pure offline)
- **Data**: 1M parsed human Showdown replays (`parsed-replays` v?)
- **Reward**: shaped per-step (specific function not in gin)
- **Battle backend**: `poke-env`
- **Position in Metamon hierarchy**: paper-era baseline. The "L" in their
  ablation suite (Small/Medium/Large × IL/RL/Synthetic).

### MediumRL_Aug (50M params)

- **Source**: paper era (RLC 2025)
- **Architecture**: AMAGO multi-task transformer, "medium" variant
- **Training method**: same as LargeRL (exp filter / AWR-style, offline)
- **Data**: 1M human replays (the "_Aug" suffix is UNDOCUMENTED in the
  files I checked — likely "augmented" data, but no explicit doc found)
- **Position**: smaller-cheaper baseline. Same family as LargeRL.

### SyntheticRLV2 (200M params)

- **Source**: paper era (RLC 2025), **September 2024 — the paper's "best"
  policy at publication time**
- **Architecture**: AMAGO **multi-task** Perceiver transformer with
  **value classification** (vs scalar value head)
- **Training method**: offline RL with **binary filter** (not exp)
  - `fbc_filter_func = binary_filter` (action: do it if advantage > 0,
    ignore otherwise — much sharper than AWR's exp weighting)
  - offline_coeff=1.0, online_coeff=0.0
- **Data**: 1M human replays + **4M diverse self-play battles**
- **Position**: "Final 200M actor-critic model with value classification"
  — peaks the paper's RLC 2025 results. Recently surpassed by
  PokeAgent-era models (Kakuna, Kadabra3, Alakazam) but still
  competitive in Gen1-4.

### Minikazam (4.7M params) — the strongest in our set

- **Source**: PokeAgent Challenge era (2025, post-paper)
- **Architecture**: **RNN** (NOT transformer) — the only RNN in our set
- **Training method**: offline RL with binary filter
- **Reward**: `AggressiveShapedReward` (custom shaped reward function)
- **Data**: `parsed-replays v4` (newer dataset than paper-era) + **~5M
  self-play battles from Alakazam** (the strongest model at the time)
- **Position**: "An attempt to create an affordable starting point for
  finetuning." Tries to compensate for low parameter count by training
  on Alakazam's strong-opponent self-play data.

**Why is the SMALLEST model the STRONGEST in our set?**
- Trains on data from Alakazam (a 57M model that was the best gen9ou
  agent during PokeAgent Challenge) → effectively distilling a stronger
  policy
- Uses newer parsed-replays v4 (better data quality than paper-era)
- PokeAgent-era methodology improvements (the README says these reduced
  model sizes, reduced reward shaping emphasis, improved generalization)

## Models NOT in our pool — what we're missing

Per the Metamon README, these are the **actually-strong** public models
that we DON'T benchmark against:

| Model | Size | Date | Why notable |
|---|---|---|---|
| **Abra** | 57M | Jul 2025 | Best gen9ou agent open-sourced during PokeAgent Challenge |
| **Kadabra3** | 57M | Sep 2025 | #1 in Gen1OU qualifier, #2 in Gen9OU (PokeAgent Challenge) |
| **Alakazam** | 57M | Sep 2025 | Final PokeAgent effort; Minikazam was distilled from this |
| **Kakuna** | 142M | Dec 2025 | **Best public Metamon model** — leads on every metric, 71% GXE on G9OU |
| **Superkazam** | 142M | 2025 | Precursor to Kakuna |

**Comparing our results vs MMs we have selected effectively measures
"are we as good as 2024 paper-era baselines + Minikazam." It does NOT
measure "are we elite in absolute terms"** — Kakuna at 71% Gen9OU GXE on
the actual ladder is the current public ceiling, and we're not testing
against it.

## Training methodology differences (Metamon vs us)

| Aspect | Metamon | Us |
|---|---|---|
| RL framework | **AMAGO** (offline RL, transformer + per-step reward) | **PPO** (online RL + BC anchor + AWR rehearsal) |
| Data | Parsed Showdown replays (1M-4M+) + self-play | Our memmap (`human_v8_100k`, ~200k filtered eps) |
| Reward | Shaped per-step (DefaultShapedReward / AggressiveShapedReward) | Sparse terminal (--reward-style terminal) |
| Architecture | AMAGO Perceiver multi-task transformer or RNN | Our spatial(6L,256d) + temporal(4L,512d) transformer |
| Offline-vs-online | Pure offline (offline_coeff=1.0) | Pure online (PPO collect → update loop) |
| Filter function | exp_filter (AWR) OR binary_filter | N/A (we use PPO + BC anchor + AWR rehearsal as separate components) |
| Self-play composition | Curated self-play datasets (e.g., 4M battles with diverse opps) | Continuous self-play during training |
| Optimizer | lr=1.5e-4, warmup=1000, l2=1e-4, grad_clip=1.5, AdamW | lr=8e-5, warmup_iters=5, AdamW |
| Battle backend | `poke-env` (paper era) or `pokeagent` (PokeAgent era custom parser) | `poke-env` |

**The key methodological gap**: Metamon is **fundamentally offline RL**
on curated datasets. We're **fundamentally online RL** continuously
generating fresh data via self-play + externals. Different problem
formulations even though they target the same task.

The Metamon team explicitly state in their README:
> "Broadly speaking, we *reduced* model sizes, reward shaping, and the
> paper's emphasis on long-term memory while *improving* generalization
> over diverse team choices and prioritizing support for gen9ou. However,
> it took several iterations to recover the paper's Gen 1-4 performance."

So even Metamon's own evolution was: bigger models with shaped rewards
→ smaller models with simpler rewards and more diverse teams. Our
PPO + syn-teams approach is in spirit closer to PokeAgent-era Metamon
than the paper era.

## What this means for our results

1. **Beating LargeRL/MediumRL/SynthRL** = matching 2024 paper-era
   baselines (Aug-Sep 2024). Necessary, not sufficient.
2. **Beating Minikazam** = a real signal, because Minikazam was
   distilled from Alakazam (a stronger 57M model). Currently we lose at
   16% on MC teams (snap_0139). Getting Mini to >40% would suggest
   reaching PokeAgent-era strength.
3. **We do NOT have data on our standing vs Kakuna/Kadabra/Alakazam/Abra**
   — the actually-strong PokeAgent-era models. Without those evals, we
   can't claim "elite" status.
4. **Our methodology is different enough from Metamon's** that direct
   comparison of architectures/approaches isn't apples-to-apples. They
   succeed via offline RL on massive curated datasets; we succeed (or
   not) via online PPO on a smaller, faster-iterating loop.

## Possible follow-up actions

(All low priority; informational reference, not blocking work)

1. **Add Kakuna + Kadabra3 + Alakazam to eval-only opponent pool** — not
   training (their parameter counts would crush our pod budget if used as
   training opponents). Just for H2H eval at end of run, to know where we
   actually stand on the elite scale.
2. **Try a Mini-distillation phase** — collect self-play data from our
   best snapshot vs strong opponents, then offline-finetune. Similar to
   how Minikazam was distilled from Alakazam.
3. **Investigate `parsed-replays v4`** — newer than the dataset our BC
   v10 was trained on. Could be a BC v11 candidate if quality is
   demonstrably better.

## References

- [Metamon paper (RLC 2025)](https://arxiv.org/abs/2504.04395) —
  "Human-Level Competitive Pokémon via Scalable Offline RL and Transformers"
- `metamon_ref/README.md` — official Metamon docs
- `metamon_ref/metamon/rl/pretrained.py` — model class definitions
- `metamon_ref/metamon/rl/configs/training/*.gin` — training hyperparameter configs
- `metamon_ref/metamon/agents/` — agent class implementations
- [Metamon project website](https://metamon.tech)
- [PokéAgent Challenge](https://pokeagent.github.io)
