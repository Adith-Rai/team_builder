# Technical Architecture

## Battle Policy Network

### Observation Space
The observation is a flat float vector encoding the full visible battle state.

**Components (per the current features.py):**
- Active Pokemon (325 dims each): species k-hot(128), type one-hot(19), HP%(1), status(7), boosts(7), item k-hot(64), ability k-hot(64), volatile statuses(15), tera type(20)
- Bench Pokemon (120 dims per side): 5 slots × [species k-hot(16) + HP%(1) + status(7)]
- Move type histograms (38 dims): 19 types × 2 actives
- Board state (43 dims): hazards with layers, screens/veil, TR, weather(5), terrain(5), mechanics(12)
- Matchup scalars (3): SE availability for both sides + party SE count
- Moved-first heuristic (3): speed comparison with Tailwind/PAR/TR
- Turn context (1): normalized turn number

**Current config (from features.py):**
- N_TYPES = 19
- H_SPECIES = 128 (hash buckets for species)
- H_MOVE = 128 (hash buckets for moves)
- H_ITEM = 64 (hash buckets for items)
- H_ABILITY = 64 (hash buckets for abilities)
- K_SPECIES = 2, K_ITEM = 2, K_ABILITY = 2 (number of hash functions)
- MAX_BENCH = 5
- N_VOLATILE = 15 (key volatile statuses per active)
- Total obs dimension: 978 floats (3,912 bytes)

### Action Space
**Singles (current):**
- 4 move slots + 5 switch slots = 9 actions
- Legal mask: binary vector indicating which actions are available
- Modifier flags: tera, mega, z-move, dynamax (binary per-action)

**Doubles (future):**
- Per-Pokemon action selection (2 active Pokemon)
- Move targeting (select target for each move)
- Requires architectural changes

### Network Architecture (policy_heads.py - BattlePolicy)
```
Input: obs vector (flat float)
  |
  v
[MLP Encoder] -- configurable layers, ReLU
  |
  +-- [Step Type Embedding] (optional)
  +-- [Context Extra Projection] (optional)
  |
  v
[LSTM Core] (optional, configurable hidden size)
  |
  v
[Action Head] -- 9-way logits (4 moves + 5 switches)
  or
[Hierarchical Head] -- move-vs-switch (2-way) + move (4-way) + switch (5-way)
  |
  +-- [Value Head] -- scalar state value
  +-- [Move-First Auxiliary Head] -- binary prediction
  +-- [Modifier Heads] -- per-modifier binary (tera, dmax, etc.)
```

### Per-Slot Encoders (optional)
When enabled, each move slot and switch slot gets its own feature encoding:
- Move slots: move hash + type + category + power + accuracy + PP info
- Switch slots: species hash + type + HP + status
- These are concatenated with the core representation before the action head

## Training Pipeline

### Phase 1: Data Generation
```
Showdown Server (Docker) <--> poke-env <--> Observer (RecorderMixin)
                                              |
                                              v
                                         JSONL files
                                    (one line per timestep)
```

**JSONL schema per line:**
```json
{
  "episode_id": "battle-gen9ou-NNN::our",
  "t": <turn_number>,
  "obs": [<float_vector>],
  "legal": [0/1 x 9],
  "action": <int 0-8>,
  "done": <bool>,
  "winner": <"our"|"opp"|null>,
  "result": <1.0|-1.0|0.0|null>,
  "modifiers": {"tera": <0/1>, ...},
  "move_slots": [<per-slot features>],
  "switch_slots": [<per-slot features>],
  "ctx_extra": [<context features>]
}
```

### Phase 2: Behavioral Cloning Training
```
JSONL files --> StreamingEpisodeDataset --> DataLoader
                                              |
                                              v
                                    BattlePolicy (forward)
                                              |
                                              v
                                    Loss: masked_policy_ce + modifier_bce
                                              |
                                              v
                                    Adam + AMP + EMA + LR scheduling
```

**Key training features:**
- Mixed precision (AMP) for speed
- Exponential moving average (EMA) of weights
- Label smoothing
- Gradient accumulation
- Legal-action-masked cross-entropy loss
- Sequence batching for LSTM (episodes grouped, padded)

### Phase 3: RL Training (planned)
```
Showdown Server <--> poke-env <--> RL Agent (BattlePolicy)
                                      |
                                      v
                               RewardShaper (potential-based)
                                      |
                                      v
                               PPO / similar update
```

**Reward shaping (rewards.py):**
- Phi = alpha * (opp_fainted - our_fainted) + beta * (our_HP% - opp_HP%)
- Shaped reward = gamma * Phi(s') - Phi(s) + raw_reward
- Tempo tax per decision (encourage faster wins)
- Clipped accumulated reward

## Infrastructure

### Docker Services (docker-compose.yml)
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| showdown | Node.js (built from source) | 8000 | Pokemon Showdown server |
| trainer | CUDA 12.1 + Python 3.11 | - | Training & inference |
| tensorboard | TensorBoard | 6006 | Training visualization |

### Dependencies (requirements.txt)
- poke-env==0.10.0
- torch==2.3.1 (CUDA 12.1)
- numpy, pandas, tqdm, tensorboard

## Data Sources

| Source | Path | Purpose |
|--------|------|---------|
| Smogon usage stats | raw_data/pokemon_usage/ | Team building, metagame analysis |
| Bulbapedia items | raw_data/items/ | Item database |
| Move data | raw_data/movesets/ | Move definitions, learnsets |
| Pokemon data | raw_data/pokemon/ | Species stats, forms, abilities |
| PBS files | raw_data/pokemon/PE20, ScarletViolet_PBS | Complete Pokemon data |
| Self-play JSONL | pokemon-ai-starter/pokemon-ai/data/datasets/obs/ | BC training data |

## Known Technical Debt

1. **Observer protocol parsing**: Uses fallback attribute names, O(T^2) log scanning, hardcoded protocol format strings
2. **Feature hashing**: Hash bucket sizes and number of hash functions chosen without clear justification
3. **Global state in training**: `collate()` functions rely on global `args` variable
4. **Action space hardcoded**: 9 actions (4+5) embedded in multiple places; not configurable for doubles
5. **No format abstraction**: Everything assumes Gen 9 OU singles
6. **Docker-only**: Showdown URL hardcoded to Docker container name
