# METAMON_LEARNINGS.md

**Created:** 2026-04-22 (Session 37)
**Status:** Research doc. Informs — does not replace — `NEXT_SESSION.md` priorities.
**Source:** Read `metamon_ref/` directly. All claims cite file paths + line numbers.

This is a concrete Metamon-vs-Ours architectural comparison written after a deep read of
the metamon reference repo. **We do NOT propose copying their code.** We extract the
design principles, judge which transfer to our Gen9 OU setup, and rank what's worth
trying next.

Context: Their 4.7M GRU ("Minikazam") sits at SR 1429 on PokeAgent, our 13.38M
transformer sits at SR 1444. They use 2.8× fewer params. Why?

---

## TL;DR — the 5 principles to take away

1. **Temporal model > spatial model in Metamon.** Even Minikazam (4.7M total) puts most
   of its capacity in the temporal GRU (400d/2L), not the per-step encoder (64d/3L
   Perceiver). At Large scale the ratio is 8:1 temporal:spatial d_model. **Ours is 1:1.**
2. **Per-step encoder is a bottleneck, on purpose.** Perceiver (Minikazam) or VIT-style
   scratch tokens (Small/Medium/Large) force the model to compress each turn into a small
   fixed representation. Output dim stays in 300–1760 range regardless of model size.
3. **4 independent critics + popart, not a single distributional critic.** All sizes use
   `NCritics` ensemble (4 heads for Small/Medium/Large/Minikazam, 6 for Kakuna). popart
   normalizes value targets to keep gradients stable under reward-scale shifts.
4. **Multi-gamma / multi-horizon training is an AMAGO default.** They train policies that
   can condition on different discount factors simultaneously. We train a single γ.
5. **Text tokenization is their data strategy, not a Pokemon-mechanics choice.** Treating
   the observation as ~87 words with a shared 2541-token vocab means gens 1-9 share the
   same tokenizer. Adding a gen costs ~0 architectural changes. Our structured entity
   scheme has the same property (FormatConfig already abstracts it) but our *vocab tables*
   need deliberate expansion.

**Highest-leverage candidate experiment (before any BC scaling):** shift capacity from
spatial → temporal. This is already "Option A / Exp 5" in NEXT_SESSION.md. This doc
strengthens the case: every Metamon size variant does it.

---

## 1. Architecture — verified from gin configs + model.py

### 1.1 The four size configs (direct quotes from configs)

**Minikazam (4.7M, the one we're compared against on the ladder)**
File: `metamon_ref/metamon/rl/configs/models/minikazam.gin`
```
# Per-step encoder: Perceiver
MetamonPerceiverTstepEncoder.d_model = 64
MetamonPerceiverTstepEncoder.n_layers = 3
MetamonPerceiverTstepEncoder.n_heads = 4
MetamonPerceiverTstepEncoder.latent_tokens = 5        # → output 5*64 = 320-dim / turn
MetamonPerceiverTstepEncoder.numerical_tokens = 3
MetamonPerceiverTstepEncoder.dropout = .05

# Temporal: GRU
traj_encoders.GRUTrajEncoder.n_layers = 2
traj_encoders.GRUTrajEncoder.d_hidden = 400
traj_encoders.GRUTrajEncoder.d_output = 300

# Actor / critic: 2L/256 hidden, 4 critics, popart ON
Agent.popart = True
Agent.num_critics = 4

MetamonAMAGOExperiment.max_seq_len = 64
```

**Small (15M)** — `configs/models/small_agent.gin`
```
# Per-step: TimestepTransformer (VIT-style scratch tokens)
MetamonTstepEncoder.d_model = 100
MetamonTstepEncoder.n_layers = 3
MetamonTstepEncoder.n_heads = 5
MetamonTstepEncoder.scratch_tokens = 4      # output 4*100 = 400-dim / turn
MetamonTstepEncoder.numerical_tokens = 6

# Temporal: causal transformer, big
traj_encoders.TformerTrajEncoder.n_layers = 3
traj_encoders.TformerTrajEncoder.n_heads = 8
traj_encoders.TformerTrajEncoder.d_ff = 2048        # 4× d_model
traj_encoders.TformerTrajEncoder.d_model = 512

MetamonAMAGOExperiment.max_seq_len = 200
```

**Large (200M)** — `configs/models/large_agent.gin`
```
MetamonTstepEncoder.d_model = 160
MetamonTstepEncoder.n_layers = 5
MetamonTstepEncoder.scratch_tokens = 11       # output 11*160 = 1760-dim / turn
MetamonTstepEncoder.token_mask_aug = True     # data aug enabled only here

traj_encoders.TformerTrajEncoder.n_layers = 9
traj_encoders.TformerTrajEncoder.n_heads = 20
traj_encoders.TformerTrajEncoder.d_ff = 5120
traj_encoders.TformerTrajEncoder.d_model = 1280
MetamonAMAGOExperiment.max_seq_len = 128
```

### 1.2 The temporal:spatial d_model ratio table

| Model | Spatial d_model | Temporal d_model | Ratio |
|---|---|---|---|
| Minikazam (4.7M GRU) | 64 | 400 (GRU hidden) | **6.25** |
| Small (15M) | 100 | 512 | **5.12** |
| Medium (50M) | 100 | 768 | **7.68** |
| Large (200M) | 160 | 1280 | **8.00** |
| **Ours (13.38M)** | **384** | **384** | **1.00** |

This is the architectural anomaly NEXT_SESSION.md already flags. The Metamon data shows
this is not accidental — **every size they ship uses ~5-8× temporal:spatial**. The
smallest competitive Metamon config still puts 400d of hidden state in the sequence model
on top of a 320d per-step output.

### 1.3 Spatial output dim (what the temporal model ingests)

| Model | Per-turn output dim (what feeds temporal) |
|---|---|
| Minikazam | 5 latents × 64 d = **320** |
| Small | 4 scratch × 100 d = **400** |
| Medium | 6 scratch × 100 d = **600** |
| Large | 11 scratch × 160 d = **1760** |
| **Ours** | 1 pooled summary × 384 d = **384** |

Two things stand out:
- Metamon keeps **multiple summary vectors per turn** (they flatten, not pool). Our
  `SpatialTransformer.summary_attn` in `src/model.py:332` collapses to a single 384-dim
  vector via attention pooling.
- Their per-turn output is a *decreasing* fraction of their per-step compute as models
  grow. Minikazam's 320-dim is 100% of its spatial capacity flattened. Large's 1760-dim
  is a fraction of the 160-dim × tokens internal state.

**Principle:** give the sequence model a richer per-step representation than a single
pooled vector. Our pooled summary may be a silent bottleneck — the temporal transformer
only sees one d-dim snapshot of each turn, not the structured 16-token spatial state.

### 1.4 Critic design — NCritics ensemble, not distributional

From `minikazam.gin:19-24`:
```
Agent.critic_type = @actor_critic.NCritics
actor_critic.NCritics.activation = "leaky_relu"
actor_critic.NCritics.n_layers = 2
actor_critic.NCritics.d_hidden = 256
Agent.popart = True
Agent.num_critics = 4
```

- **4 independent scalar critics** (all sizes except Kakuna which uses 6 two-hot critics)
- **popart normalization ON** across all configs
- **2-layer MLP**, leaky ReLU
- Training with soft target update: `tau = 0.004` (binary_maxq_rl.gin:6) or `tau = 0.008` (kakuna.gin:6)

**Ours:** single distributional value head, 51 bins, two-hot targets (`src/model.py:577-586`).
That design came from the ps-ppo paper. It's not wrong — ps-ppo gets >1900 Elo with it —
but Metamon proves an NCritics scalar ensemble + popart works too, and their ensemble is
the only variance-reduction mechanism they need.

For PPO specifically, an ensemble adds code complexity we probably don't want right now.
But **popart is a cheap bolt-on** that would help if our value scale drifts during
self-play (which Exp 4's collapse suggests it might).

### 1.5 Actor — same across all sizes, tiny

`MetamonMaskedActor`:
- 2-layer FFN over temporal output
- d_hidden = 256 (Minikazam/Small) / 300 (Medium) / 512 (Large)
- leaky_relu activation
- illegal_actions passed as separate input and masked at logits

**Ours:** `policy_head` is `Linear(3D → D) → GELU → Linear(D → 1)` applied per-action
with action-context broadcasting (`src/model.py:570-574`). Our action-conditioned head
is more elaborate than Metamon's. That's justified by our flat 9-way action space where
the per-action moves carry structure (BP, accuracy, type) worth attending to.

### 1.6 RL hyperparameters — direct quotes

**Kakuna (the flagship 142M recipe), `configs/training/kakuna.gin`:**
```
agent.Agent.reward_multiplier = 10.
agent.Agent.tau = .008
agent.Agent.online_coeff = 0.1        # 90% offline, 10% online self-play
agent.Agent.offline_coeff = 1.0
agent.Agent.fbc_filter_func = @agent.leaky_relu_filter    # soft filter for big models
agent.leaky_relu_filter.beta = .3
agent.leaky_relu_filter.neg_slope = .05
MetamonAMAGOExperiment.learning_rate = 1e-4
MetamonAMAGOExperiment.critic_loss_weight = 12.5          # critic loss 12.5× actor loss
MetamonAMAGOExperiment.lr_warmup_steps = 10000
MetamonAMAGOExperiment.grad_clip = 1.0
MetamonAMAGOExperiment.l2_coeff = 1e-4
```

**Binary+MaxQ (the small-model recipe), `configs/training/binary_maxq_rl.gin`:**
```
agent.Agent.tau = .004
agent.Agent.online_coeff = 0.25       # 75% offline, 25% online
agent.Agent.fbc_filter_func = @agent.binary_filter    # hard {0,1} filter
MetamonAMAGOExperiment.learning_rate = 1.5e-4
MetamonAMAGOExperiment.critic_loss_weight = 10.
MetamonAMAGOExperiment.lr_warmup_steps = 1000
MetamonAMAGOExperiment.grad_clip = 1.5
```

Two things jump out:
- **critic_loss_weight = 10–12.5.** In our PPO we use `vf_coef=0.5` (standard). Metamon's
  critic gets an order of magnitude more gradient than the actor. This is offline RL so
  not directly comparable — their critic needs to be accurate for the advantage filter
  to mean anything — but it's a reminder that value-function accuracy is load-bearing.
- **Reward multiplier = 10.** Binary terminal reward is scaled up before training. With
  popart that's a no-op in theory, but it affects filter thresholds and Q-value scale.

### 1.7 Loss — Binary+MaxQ flavor

Offline actor loss (`binary_maxq_rl.gin:13`):
```
w(h, a) = binary_filter(A(h, a)) = 1 if A > 0 else 0
L_actor = -w(h,a) * log π(a|h) - λ * E_a'[Q(h, a')]   # FBC with MaxQ regularizer
```

This is **not directly applicable to our PPO setup** — we're on-policy, we don't filter
by advantage sign, we use clipped surrogate loss. But it informs the next paragraph.

---

## 2. State representation — where they diverge most from us

### 2.1 Text + numerical multimodal sequence

File: `metamon_ref/metamon/il/model.py:67-98`
```python
class MultiModalEmbedding(nn.Module):
    def __init__(self, token_emb_dim, numerical_d_inp, output_dim,
                 numerical_tokens, dropout):
        self.text_emb = nn.Linear(token_emb_dim, output_dim)
        self.num_emb = nn.Linear(numerical_d_inp, numerical_tokens * output_dim)
    def forward(self, text_emb, numerical_features):
        text_emb = F.leaky_relu(self.dropout(self.text_emb(text_emb)))
        num_emb = F.leaky_relu(self.dropout(self.num_emb(numerical_features)))
        num_emb = rearrange(num_emb, "b l (l2 d) -> b l l2 d", l2=numerical_tokens)
        seq = torch.cat((text_emb, num_emb), dim=-2)
        return seq
```

Per turn the input becomes `~87 text tokens + 3-6 numerical tokens = 90-93 tokens`
which the per-step encoder compresses.

**Compare to ours:** we have 14 entity tokens (field, transition, 6 our mons, 6 opp mons)
plus 2 decision tokens = 16 tokens, each at 384d. Structure is explicit: one token per
Pokemon. In Metamon, structure is implicit in word ordering: "`<player>` Charizard
Leftovers Blaze fire flying ..." — the model learns slot semantics from position.

**Verdict for us:** keep entity tokenization. It's the ps-ppo breakthrough we already
captured. Metamon's text approach is a scaling decision (one tokenizer for gens 1-9) and
a dataset decision (text parses cleanly from Showdown replay logs). Our structured
features carry more Pokemon-specific inductive bias per dim.

### 2.2 Tokenizer size

`metamon_ref/metamon/tokenizer/DefaultObservationSpace-v1.json` has **2541 tokens**.
Covers: all species across gens 1-9, all moves, all items, all abilities, all natures,
plus special markers (`<move>`, `<switch>`, `<opponent>`, `<conditions>`, etc.).

**Our vocab sizes** (`src/model.py:60-63`):
```
n_species: 1548    # needs verification for gen 6-9 coverage
n_moves: 953
n_items: 2340
n_abilities: 314
```

These are separate embedding tables per field. Total "tokens" is 1548+953+2340+314 = 5155
across 4 distinct vocabularies. For multi-gen prep we need to confirm each table covers
gens 4-9 species/moves/items/abilities. Metamon's single-vocab approach is implicitly
multi-gen — they didn't have to "add gen 9"; they just extended the one vocab.

### 2.3 Previous-action features

Ours (`src/model.py:265-288` `TransitionNet`):
- our_action (id), opp_action (id), TRANSITION_CONT_DIM continuous features

Metamon (`interface.py` DefaultObservationSpace, §2 of the Explore agent's report):
- `<player_prev>` + previous player move words (3)
- `<opp_prev>` + previous opponent move words (3)

Both systems encode previous actions. Ours uses structured embeddings + continuous flags
(SE / resisted / immune / crit per ps-ppo). Theirs uses text. **Neither is clearly better;
we already have this.**

---

## 3. Data pipeline — the underrated differentiator

### 3.1 Scale

- **4M+ parsed human replay trajectories** (`metamon-parsed-replays` HF dataset, v5)
- **18M+ self-play trajectories** added on top (`metamon-parsed-pile`)
- Each battle yields 2 perspectives (POV reconstruction)

Ours: we have human_v8 / memmap_v8 (smaller scale, stale dims per our MEMORY.md) plus
~5-6M states of self-play collected during PPO. **Gap: ~3-5× less human BC data, ~20×
less self-play data.**

### 3.2 POV replay reconstruction (their "secret sauce")

Files: `metamon/backend/replay_parser/parse_replays.py` + `forward.py` + `backward.py`
(not read in this session; Explore agent's summary): Showdown replays are spectator-view
(full opponent info). Metamon's parser runs a forward pass that tracks "what each player
could actually see" at each turn, masking out unrevealed opponent info. This turns 1
replay → 2 POV trajectories and makes human replay data usable for BC (previously this
was why Pokemon BC underperformed: the model saw info the player shouldn't have had).

**We do not currently do this.** Our BC dataset was bot-vs-bot self-play (which is
already POV-consistent because bots only see what's observable). For scaling to human
replays — which is where the BC ceiling gets lifted — we'd need this or equivalent.

### 3.3 BC training recipe (`il/train.py`)

```
Batch size: 48
Learning rate: 1e-4, cosine annealing to eta_min=1e-4 (= constant LR effectively)
Optimizer: Adam with weight_decay=1e-4
Gradient clip: 2.0
Dropout: 0.05
Epochs: 500 max, early stopping patience=2
Loss: F.cross_entropy with ignore_index for illegal/unavailable actions
Metrics: top-1 and top-2 accuracy
Train/val split: 90/10 with fixed seed
```

**Notable differences from our train_bc.py:**
- Their LR is LOWER (1e-4 vs our ~3e-4 typical). Their schedule is effectively flat
  (eta_min = 1e-4 = LR itself).
- Dropout **0.05** (vs our default **0.1**).
- Batch size **48 sequences** (variable-length), so effective tokens per batch depends on
  episode length. Max seq len caps at 200 turns for transformers, 64 for Minikazam GRU.
- Pure cross-entropy, no auxiliary losses. No value loss during BC.

**For us:** if we do the multi-gen BC retrain, trying dropout=0.05 and a flat LR schedule
(with short warmup) costs nothing. The bigger thing is the DATA quantity gap, not the recipe.

---

## 4. Minikazam vs Kakuna — what scales

`metamon_ref/README.md` (not read this session but inferable from configs + `rl/pretrained.py`):

- Minikazam (4.7M GRU, `small_rnn.gin`): our ladder peer
- Small Agent (15M transformer): paper baseline
- Large Agent (200M transformer): paper best
- Kakuna (142M transformer, `kakuna.gin` training recipe): post-paper best public model;
  uses `leaky_relu_filter` (soft advantage) instead of `binary_filter`, 6 critics, higher
  reward multiplier

**Observation:** the step from Minikazam → Small is a 3× param jump for what appears to
be a modest ladder gain (SR 1429 vs ~similar range). The step from Small → Large is 13×
for a reported ~60% → 80% GXE jump. **Scaling returns exist but are not linear.**

At our size (13.38M) we are ~2× Minikazam / ~0.9× Small. The fact that a 4.7M GRU nearly
matches us suggests:
1. Our architecture is less parameter-efficient per Elo than Metamon's smallest config.
2. OR our training signal is weaker (less data, worse filter, less exploration).
3. OR both.

The diagnosis from §1: **we over-invest in spatial capacity relative to temporal capacity.**
A 384d/4L/4H spatial encoder with entity tokens has ~40% of our 13.4M params.

---

## 5. What to apply to our project

### 5.1 Strong yes (high confidence, low cost)

**A. Keep entity tokenization.** ps-ppo's proven pattern + Metamon's text approach is a
scaling trick, not a quality trick. Ours is denser-per-dim for Pokemon mechanics.

**B. Confirm architectural plan: shift capacity spatial → temporal.** NEXT_SESSION.md
Option A already proposed spatial 256d/3L + temporal 512d/3L. Metamon data supports this
for *every* size class. **Target ratio ~2:1 temporal:spatial d_model minimum**, not
inverted like now.

**C. Trial popart in the critic.** Add running mean/std normalization for value targets.
Small implementation cost. Would guard against the Exp 4 value-scale drift that we know
correlated with collapse. Not urgent — safeguards (adaptive entropy + early stop) already
cover the collapse pathway — but a defensive addition.

**D. Drop dropout to 0.05 for next BC retrain.** Metamon's choice, ps-ppo also uses low
dropout. Our 0.1 may be excessive for a model that's under-fitting (BC plateau at ~22-26%
is an under-fitting signal).

### 5.2 Moderate yes (high upside, significant cost)

**E. Multi-summary temporal input.** Currently `SpatialTransformer` outputs one pooled
384d vector per turn (`src/model.py:386-391`). Metamon Small outputs 4 × 100 = 400d;
Large outputs 11 × 160 = 1760d flattened. Change our output from one `summary` vector to
K ≥ 2 "summary tokens" (can be actor_out + critic_out + field + pooled, or K learned
scratch tokens à la Metamon). Temporal transformer ingests these per-turn as K × d_model
after positional encoding. Code change is moderate. This is a distinct lever from "make
temporal bigger."

**F. POV replay reconstruction for human data.** Biggest-lift item but the most important
if we ever want to scale BC. Their parser is open-source. For multi-gen we may need this
anyway (Showdown replays are the only source of gen6-8 data at scale). Park this until
after multi-gen vocab work.

### 5.3 Ambivalent (worth noting, not acting on now)

**G. 4-critic ensemble (`NCritics`)** — for PPO it's not a natural fit. Our 51-bin
distributional head already provides richer signal than scalar regression. Skip.

**H. Multi-gamma training (AMAGO feature)** — their infra supports it for free; ours does
not. Retrofit cost is high for unclear benefit on a single format.

**I. Text tokenization instead of entity tokens** — clear downgrade for per-dim expressiveness;
only attractive if we wanted easy multi-gen transfer and we've already decided to do
multi-gen through FormatConfig + expanded vocab tables, which is equivalent in end state.

### 5.4 No (decided against)

**J. Copy AMAGO framework.** It's a sequence-RL framework (off-policy, Q-learning). Our
on-policy PPO is a fundamentally different training regime. Switching would invalidate
everything learned. The user has also explicitly prioritized PPO-based work.

**K. Offline RL (Binary+MaxQ) to replace PPO.** Would require retraining the entire
pipeline. The advantage filter idea is worth keeping in mind for a future experiment on
offline data from our own self-play, but it's not a PPO drop-in.

---

## 6. Recommended sequencing (integrates with existing NEXT_SESSION plan)

Current plan (NEXT_SESSION.md §"What to do next"):
1. IMMEDIATE: Study Metamon ← **this doc completes that step**
2. Multi-gen prep (vocab + features + team_generator + format_config)
3. After multi-gen: Option A (capacity reallocation) / B (BC scaling) / C (MCTS) / D (PokeAgent)

**Adjusted recommendation:**

1. **Still do multi-gen prep next.** Reason unchanged: doing BC-scaling or capacity
   changes now, then adding multi-gen later, means redoing the BC. One-pass ordering wins.
2. **When you do capacity reallocation (Option A), include multi-summary temporal input
   (item E above).** Single code PR: new spatial config + new `n_summary_tokens` option.
   The experiment tests BOTH "bigger temporal" AND "richer per-turn signal" at once.
3. **Lower BC dropout (0.05) and use a warmup + cosine schedule on the multi-gen BC.**
   Low-cost, Metamon-endorsed.
4. **Consider popart as a defensive add for the next PPO run post-retrain.** Not
   blocking; can be tried independently.
5. **POV replay reconstruction comes AFTER we've validated that more BC data helps** —
   i.e. after running multi-gen BC once with whatever data we can already scrape, and
   checking the resulting Elo delta. Only if more human data is clearly the next lever do
   we invest in the parser work.

---

## 7. Open questions this reading couldn't answer

1. **Does popart actually help in our specific PPO setup?** Metamon uses it in offline
   Q-learning where value scale drift is worse. Untested claim in PPO-self-play.
2. **Does the "multiple summary tokens" pattern survive ablation?** Metamon's scratch
   token count scales with model size (4 → 6 → 11). Is that a scaling requirement or a
   free hyperparameter? No ablation in the paper abstract we've read.
3. **Does Minikazam's GRU beat an equivalent-param transformer at that scale?** Unknown
   — they don't publish a transformer-at-4.7M config to compare.
4. **What is the Gen9 OU GXE of Minikazam specifically?** The 71% claim in Explore's
   report is for Kakuna. Minikazam's Gen9 performance would tell us how much of the
   "they tie us at 4.7M" story is architecture vs just experience at Gen9.

Worth investigating if/when we run into a blocker that these would answer.

---

## 8. Files to re-read if extending this analysis

- `metamon/rl/train.py` — full online-offline RL loop (we read configs only)
- `metamon/backend/replay_parser/{forward,backward}.py` — the POV reconstruction trick
- `metamon/data/parsed_replay_dset.py` — data filters + dataset structure
- `metamon/rl/self_play/` — opponent pool composition (relevant if we change PFSP)
- `metamon/baselines/` — not opened; may contain heuristic-bot code we could crib

---

**End of Metamon learnings. Next concrete action per NEXT_SESSION.md:** multi-gen vocab
expansion (vocab.py + features.py + team_generator.py + format_config.py). This doc should
be revisited before Option A (capacity reallocation) to make sure the design actually
bakes in the spatial:temporal rebalancing and multi-summary temporal input.
