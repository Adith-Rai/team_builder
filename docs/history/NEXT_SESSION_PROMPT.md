# Next Session Prompt

Copy-paste this to start the next session:

---

We're building a Pokemon AI that battles in Gen9 OU. Read `docs/NEXT_SESSION.md` top-to-bottom — it's the canonical handover document. Then read `docs/RESEARCH.md` §0 for architecture context.

**Where we are:** 13.38M model at Elo 1058 (internal), Skill Rating 1444 on the PokeAgent Challenge ladder (rank #12, above Metamon's 4.7M Minikazam but below their 15M model). Architectural ceiling confirmed — hyperparameter tuning is exhausted.

**What to do this session (in order):**

1. **Study Metamon's architecture** in `metamon_ref/` (cloned, read-only reference). Their 4.7M model nearly matches our 13.38M — we need to understand WHY. Look at:
   - Temporal vs spatial capacity ratio (we're 1:1, they're 5-8:1)
   - Entity token handling
   - BC data pipeline and preprocessing
   - Minikazam (4.7M) architecture specifically
   DO NOT copy their code. Learn the design principles, apply to our architecture.

2. **Plan capacity reallocation** based on findings. Current: spatial 384d/4L + temporal 384d/2L. Proposed direction: shrink spatial, grow temporal. But use Metamon's actual numbers as reference, not our guess.

3. **Multi-gen vocab prep** — expand species/move/ability/item tables for gens 6-9. Architecture is already gen-agnostic. Changes needed in: vocab.py, features.py (volatile effects), team_generator.py (per-gen), format_config.py (gen6-8 configs).

4. **Optionally: set up Metamon as a training opponent.** Download their Gen9 model, wrap as poke-env Player, add to our self-play pool. Training against genuinely different opponents could break the plateau.

**Key files:** `src/model.py` (architecture), `src/features.py` (entity tokenization), `src/train_rl.py` (training loop with safeguards), `src/pokeagent_submit.py` (ladder submission). All safeguards implemented: `--adaptive-entropy --early-stop --win-rate-mode ema`.

**Team selection results** in `src/team_selection_results.json`. TEAM_T is our best on the real ladder (Skill Rating 1444). TEAM_AU is second (1376). Use TEAM_T for any competitive play.

**Do not re-run hyperparameter experiments.** The ceiling at Elo 1058 is confirmed across 5 experiments and 1891 Elo matchups. The next breakthrough requires architectural changes informed by Metamon's design.

---
