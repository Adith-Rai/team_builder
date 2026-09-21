# Pokemon AI Battler - Master Project Plan

## Vision

Build a Pokemon AI system that can:
1. **Battle competitively** across all major formats (1v1, 3v3, 6v6 singles; 2v2 VGC, normal doubles, 3v3 doubles)
2. **Use any Pokemon** from the full roster - fundamentally understand battling, not just memorize matchups
3. **Build excellent teams** - choose Pokemon, items, moves, EV/IV spreads, abilities; build around given Pokemon constraints
4. **Compete on the Showdown ladder** once skill reaches human-expert level

---

## Phases

### Phase 1: Data Generation (Simulation)
**Goal:** Generate high-quality battle recordings via local Showdown self-play.

- Run Pokemon Showdown locally (currently via Docker)
- Simulate thousands of battles between heuristic bots and trained agents
- Record complete battle state as a human player would see it (all visible info)
- Store observations in JSONL format for training
- **Formats:** Start with Gen 9 OU singles, expand to other formats later

**Key Requirements:**
- Observer must capture the COMPLETE battle environment as a player sees it:
  - Own team: species, HP, status, moves (with PP), item, ability, stats, boosts, volatile conditions
  - Opponent team: species, HP (%), status, known moves, known item, known ability, revealed info
  - Field state: weather, terrain, hazards (per side), trick room, tailwind, screens, etc.
  - Turn context: who moved first, what happened, tera/mega/z-move/dmax usage
  - Legal actions: which moves and switches are available
- Observer must use CORRECT Showdown variable names and mechanics (not guessed ones)

### Phase 2: Behavioral Cloning (Observation Learning)
**Goal:** Train a model to imitate strong play from recorded battles.

- Model observes recorded battles and learns move/switch/modifier decisions
- Architecture: Recurrent (LSTM) policy network with:
  - Observation encoder (MLP)
  - Optional LSTM core for sequence modeling
  - Action head (4 moves + 5 switches = 9 actions for singles)
  - Modifier heads (tera, mega, z-move, dynamax)
  - Value head (for future RL)
- Training: Cross-entropy on expert actions with legal-action masking
- Validation: Win rate vs heuristic bots

### Phase 3: Beyond Behavioral Cloning
**Goal:** Surpass BC ceiling and develop novel strategies.

**Phase 3a: Online RL (PPO) — ATTEMPTED, PLATEAUED**
- PPO with BC warm start, KL penalty, curriculum, dense rewards, self-play, adaptive weights
- Best result: 28% avg vs strong bots (RL v2, iter 130)
- Plateaued after v1→v2→v3 iterations — fundamental issues with on-policy RL for Pokemon:
  - Websocket collection bottleneck (~30s/100 games)
  - Reward sparsity (win/loss at turn 30-80)
  - High variance (±15-20% noise in 20-game evals)
  - Sample inefficiency (5000 steps used for 4 epochs then discarded)

**Phase 3b: Offline RL (v5 — RECOMMENDED NEXT)**
- Use 5M expert demonstrations directly as training data
- No Showdown needed during training — pure GPU workload, cloud-friendly
- Candidate algorithms (ordered):
  1. Advantage-Weighted BC (simplest: weight BC loss by game result)
  2. Decision Transformer (sequence modeling with return conditioning)
  3. Implicit Q-Learning (conservative Q-function from offline data)
- Train on cloud spot instances for larger models (10-50M params)
- Separate concerns: data gen (local CPU) / training (cloud GPU) / eval (local CPU+GPU)

### Phase 4: Live Ladder Play
**Goal:** Connect to Showdown servers and battle real players on the ladder.

- Implement Showdown websocket client for live play
- Use poke-env's online battle capabilities
- Start in lower-skill formats, graduate upward
- Monitor and log performance
- Continuous learning from live games (optional)

### Phase 5: Team Building
**Goal:** AI-generated competitive teams.

- **Separate model(s)** from the battler
- Capabilities:
  - Build a full team from scratch for a given format
  - Build a team around 1-3 given Pokemon
  - Select optimal: Pokemon, moves, items, abilities, EV/IV spreads, natures
  - Account for metagame (usage stats, common threats, synergy)
- Training data: Smogon usage stats, competitive team databases, tournament results
- Approach: Likely a combination of:
  - Statistical modeling (teammate/counter correlations from usage data)
  - Learned evaluation (team strength predictor trained on win rates)
  - Search/optimization (genetic algorithm or beam search over team space)

---

## Architecture Overview

```
team_builder/
  docs/                     # Project documentation (this directory)
    PROJECT_PLAN.md         # This file - master plan
    ARCHITECTURE.md         # Technical architecture details
    STATUS.md               # Current project status
    COMMANDS.md             # Key commands and workflows
  pokemon-ai-starter/
    pokemon-ai/
      src/                  # Main source code
        observer.py         # Battle observation & JSONL recording
        features.py         # Feature extraction (obs vector + action masks)
        policy_heads.py     # Neural network architecture
        bc_train.py         # Behavioral cloning training
        bc_policy_player.py # Inference player (uses trained model)
        eval_bc_vs_bots.py  # Evaluation harness
        policy_rulebots.py  # Heuristic bot opponents
        policy_random.py    # Random baseline
        rewards.py          # RL reward shaping
        env_wrapper.py      # Environment setup
        teams_ou.py         # Team definitions
        train.py            # Legacy training entry point
        scan_jsonl.py       # Dataset validation
      data/
        datasets/obs/       # JSONL observation files (~10GB)
        models/             # Trained model checkpoints
        logs/               # TensorBoard logs
        evaluations/        # Eval results
        replays/            # Battle replays
      scripts/              # Data processing scripts
    docker-compose.yml      # Docker setup (Showdown + trainer + TB)
    requirements.txt        # Python dependencies
  raw_data/                 # Scraped Pokemon data
    items/                  # Item database
    movesets/               # Move + learnset data
    pokemon/                # Pokemon stats, forms, PBS data
    pokemon_usage/          # Smogon usage statistics
  bans/                     # Ban lists
  stoplist/                 # Format stoplists
  *.py                      # Top-level utility scripts (scraping, data combining)
```

---

## Models Required

| Model | Purpose | Input | Output | Training |
|-------|---------|-------|--------|----------|
| **Battle Policy** | Choose moves/switches in battle | Battle state observation | Action (move/switch) + modifiers | BC then RL |
| **Team Builder** | Construct competitive teams | Format rules + optional Pokemon constraints | Full team (6 Pokemon with sets) | Usage stats + win rate prediction |
| **Team Evaluator** (optional) | Rate team quality | Team composition | Quality score | Win rate data |

---

## Format Coverage Plan

**Phase 1-3 (Singles — Gen 9):**
- Gen 9 OU (6v6 singles) - primary
- Gen 9 1v1
- Gen 9 3v3 (Battle Stadium Singles)
- Other singles tiers (UU, RU, etc.)

**Phase 3+ (Doubles — Gen 9):**
- VGC (4v4 doubles, bring 6 pick 4)
- Gen 9 Doubles OU
- 3v3 doubles formats

**Phase 4+ (Triples — Gen 9):**
- Gen 9 Triples formats (3v3 triples)

**Phase 5+ (Multi-Gen Expansion):**
- Gen 4 (DPP) — Singles, Doubles
- Gen 5 (BW) — Singles, Doubles, Triples
- Gen 6 (XY) — Singles, Doubles, Triples
- Gen 7 (SM) — Singles, Doubles
- Gen 8 (SwSh) — Singles, Doubles, VGC

**Adaptation Strategy:**
- Core battle understanding transfers across formats and generations
- Format-specific heads or fine-tuning for action space differences
- Doubles requires modeling 2 active Pokemon per side, different action space
- Triples extends doubles with 3 active per side + position-based targeting
- Multi-gen requires per-gen move/ability/item databases and mechanics knowledge

---

## Key Technical Decisions

1. **Showdown as ground truth**: Clone the actual Showdown repo to reference correct mechanics, variable names, and battle logic
2. **poke-env as interface**: Python wrapper for Showdown battles (currently v0.10.0)
3. **PyTorch for models**: CUDA 12.1 support, AMP training
4. **Docker for infrastructure**: Showdown server + training containers
5. **JSONL for data**: Streaming-friendly, episode-grouped observations
6. **Observation = what a human sees**: No hidden information, complete visible state
7. **Multiple specialized models**: Battler and team builder are separate systems

---

## What Needs Fixing (Current State)

See [STATUS.md](STATUS.md) for detailed current status.

### Critical Issues
1. **Observer uses wrong Showdown variables** - tries multiple attribute name fallbacks instead of using correct poke-env API
2. **Feature extraction may be incorrect** - hash buckets, arbitrary dimensions without documentation
3. **Code is disorganized** - dead code, duplicate code, notebook-style scripts mixed with modules
4. **LSTM/non-LSTM code mixed inconsistently** - training supports both but validation is implicit
5. **10GB of observation data is likely bad** - generated from broken observer, should be regenerated after fixing

### Organization Issues
1. Duplicate files between `src/` and `src-backup/`, and between root and `scripts/`
2. Dead file: `bc_train_ddp-unused.py`
3. No README, no proper project documentation
4. Hardcoded Docker URLs in code
5. Magic numbers throughout codebase
6. Code duplication between policy_random.py and policy_rulebots.py
