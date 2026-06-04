# Plateau Hypothesis + Brainstorm

S68 (2026-06-03). Captures discussion log for plateau analysis + experiment
queue. Memory pointer: `memory/project_plateau_hypothesis_and_experiments.md`.

## Story so far

We hit a consistent 70-74% smart_avg ceiling across all configs (LR=1e-5,
3e-5, 8e-5, 1e-4, dense/terminal, prod/dev, fishbowl_v2/resume). Initially
thought ceiling was on our side. S68 MM eval reframed this:

1. **LargeRL/SyntheticRLV2/MediumRL_Aug also cap at ~67-71% vs smart bots.** Our top
   model (POST_INIT_iter139 = 1178.4 ladder) is at MM tier on metamon-competitive
   teams (50/50 vs LargeRL, 56% vs MediumRL_Aug, 49% vs SyntheticRLV2). So
   "we hit 70-74%" = "we and major metamon models share the same ceiling
   against heuristic bots."
2. **Minikazam transcends the ceiling** (92% vs bots, 84-16 over us on
   metamon-competitive). But procedural-team eval shows Minikazam dropping to
   70-30 (gap shrinks from 84% to 70%). Suggests Minikazam is **partially
   team-overfit** to the 16 curated competitive teams, not pure raw skill.

## User's hypothesis

BC v10 was trained on 1500+ Elo human replays = exposed model to elite play
patterns + team synergy. PPO 100% procedural teams = never sees teams where
those elite patterns are testable → BC patterns get NO reward signal → fade
or persist as untested noise. Model converges to "generalist with no elite
specialization."

User goal: **generalize first, but have elite capability when given elite
teams.** NOT specialize-over-generalize.

## MM training details (all of them, summary)

| Model | Arch | Train recipe | Data |
|---|---|---|---|
| LargeRL | 195M transformer | exp_rl.gin (exp-advantage offline RL) | parsed-replays-v4 (4M) |
| MediumRL_Aug | ~50M transformer | exp_rl.gin | parsed-replays-v4 augmented |
| SyntheticRLV2 | 200M synthetic | binary_rl.gin (wins-only) | Synthetic-generated (V2) |
| Minikazam | Small RNN | binary_rl.gin | parsed-replays + 5M self-play |
| LargeIL | 195M | il.gin (pure IL) | parsed-replays |

All offline (online_coeff=0.0). None do live self-play during training. They
all train on human-replay-style data. Naturally exposed to competitive team
patterns.

Our approach: BC init + online PPO + procedural teams. Different paradigm,
not necessarily worse — but loses the "trained on competitive teams" coverage.

## Brainstormed experiments — full discussion

### Complementary (layer on user's hypothesis)

**1. Replay rehearsal during PPO** — cheapest fix, directly tests hypothesis.

Mix small batch (~5-10%) of OFFLINE imitation samples from 1500+ Elo replays
into each PPO iter. Continues to expose model to elite play patterns
throughout PPO training — patterns get refined or rejected via the
co-occurring on-policy reward signal. Standard "rehearsal" trick from continual
learning literature.

Implementation:
- Pre-load replay data shards (already have HF datasets)
- Each PPO update step: mix replay batch into PPO update via auxiliary BC
  loss (small weight, ~0.05-0.1)
- Or: separate IL update step every Nth PPO step
- Cost: ~hour of code + 1 fishbowl-scale run to validate

Risk: too much rehearsal weight = converges back to BC (loses PPO benefit).

**2. Template-based team generator** (user's idea).

Define ~10-20 archetype templates (rain, weather, balance, hyper-offense,
stall, web HO, screens, sand, hazard stack, bulky offense, etc.). Each template:
- Core slots (e.g., rain template: weather setter + 2 sweepers w/ swift-swim)
- Flex slots (e.g., spinner, defogger, role-fill)
- Role requirements (must have at least 1 hazard setter, 1 wincon, etc.)
- Sample per-battle → effectively infinite variety inside each template

Implementation:
- Template definition file (YAML): list mons per slot, role requirements
- TemplateTeambuilder: yields per-battle by sampling within templates
- MixedTeambuilder: weighted mix of TemplateTeambuilder (~30%) + ProceduralTeambuilder (~70%)
- Pass via new flag `--template-teams-path templates.yaml --template-team-pct 0.3`
- Cost: ~few days of Smogon-domain template authoring + ~day of code

Risk: per-template overfit if templates too constrained. Mitigation:
templates should be inclusive (lots of flex slots).

**3. Curriculum on team complexity** — early iters simpler teams, graduate.

Standard curriculum learning. Could nest inside templates: "easy" templates
first (mono-type, fewer interactions), graduate to complex tournament archetypes.
Helps model build foundational understanding before complex synergies.

**4. Stronger BC anchor** — anchor PPO to a stronger reference instead of v10.

Current: KL=0.1 to BC v10 (trained on 1500+ replays). Could be sharpened by:
- Anchoring to a fine-tuned-on-elite-replays model (BC v11 with curated data?)
- Or to a metamon-distilled model (use one of the MMs as the KL target)

Cheap if we already have a reference checkpoint. Pulls policy structurally.

**5. Forced archetype matchups in self-play** — pair rain-vs-grass, stall-vs-HO,
etc., sometimes. Model learns archetype dynamics, not just symmetric "us vs us."

### Counter-ideas (challenge the premise)

**6. Model size is the bottleneck** — not teams.

20M params may be undersized for strategy depth. Minikazam is small but RNN;
SyntheticRLV2 is 200M. Cheap ablation: 50M model trained on current procedural
data. If it leapfrogs current 1178 Elo, size matters as much/more than teams.

**7. Encoding gaps may matter most** (Task #70 audit).

If model literally can't perceive critical info (specific item effects,
hidden ability states, opponent move predictions), no team training fixes
the perception ceiling. Foundational fix that's been on TODO list. Worth
revisiting the audit + addressing top gaps before another training arc.

**8. Reward shaping may be misaligned.**

Current dense reward: terminal + ko_coef=0.05 + hp_coef=0.02 + immune_penalty.
MM uses AggressiveShapedReward (Minikazam) or binary filter. Maybe we're
rewarding micro-things the wrong way and model optimizes wrong objective.
Worth ablation: try just terminal reward, see if it changes anything.

**9. Self-play has inherent ceiling regardless of teams.**

Self-play converges to Nash-like equilibrium where both sides plateau. To
break out may need 30-50% MM weight in pool (current 5% probably too low).
Bigger MM weight = harder learning signal early but more diverse strategy
exposure.

**10. More compute / iters might just work.**

200 iters is fairly few. Maybe 1000 iters with current config closes the gap.
Counter: bc_kl drift to >0.20 in fishbowl_v2_resume suggests we're entering
reward-hack territory, not a smooth-climb regime. More iters of current
config could MAKE things worse, not better.

### Contrarian (flag, don't pursue first)

**11. Inference-time search (PokeChamp-style).**

20M + MCTS at decision time could match 200M without search. Compute-efficient.
Decouples team knowledge from raw model scale. Most ambitious change but
potentially big leverage.

## Priority for first follow-up experiments

1. **Replay rehearsal during PPO** — cheapest direct test of hypothesis
2. **Template-team mix** — second test, addresses team-side
3. **Model size ablation** (50M run) — one ablation to rule out scale
4. **Encoding gaps fix** (foundational, long-deferred)

## External opponents from start — caveats

Discussed but constrained: MMs are way too strong as opponents from iter 0
of fresh training (model gets crushed → bad signal). Mitigations:

- Use only weaker MM variants (Minikazam too strong; would need Small* variants,
  not currently downloaded)
- Heuristic bots (smart_avg set) as easier "external opponents" early, graduate
  to MMs at mid-training
- Use MMs as small pool slice (5-10%) — what fishbowl_prod did. Could increase
  to 20-30% AFTER initial warmup iterations
- Schedule: warmup 0-20 iters pure SP, then ramp ext from 5% to 30% by iter 100

## Adjacent ideas worth tracking

- **Add ext checkpoints to era4_chain ladder** — fishbowl_prod_lr1e-4_v1 final
  (snap_0289) used externals. Adding it shows whether the "with externals"
  variant placed differently on the ladder than pure-SP variants.
- **Run external opps vs each other (MM-vs-MM matrix)** — gives a TRUE MM-tier
  Elo scale. E.g., is Minikazam dominant over LargeRL too? Or just over us?
  Worth knowing to calibrate "what tier is Minikazam at."

## Risks / open questions across all experiments

- Even with template-team mix, PPO self-play has us-vs-us — refines "exploits
  of own play on those teams," may not transfer to actual strong opponents.
  Mitigate via mixed-opponent pool.
- Curated 100-150 teams may overfit (95 games/team across 200 iters).
  Templates with infinite variety mitigate this.
- Adding MMs heavily increases compute cost (MM forward ~5x our model).
- Replay rehearsal needs careful weighting; too much = back to BC.

## Discussion log notes

- "goal is not 'beat mm' — that's a milestone. casue having elite play should cover that" (user)
- "the loose learning by bc is never tightened or properly understood" (user, articulating the BC-PPO interaction issue)
- "i forgot to mention — now that you know my idea — do you have any ideas on top of it - or counter ideas?" (user's request for brainstorming, leading to this doc)
- Several ideas above came from user pushback: template generator was user's, replay rehearsal was assistant's, both endorsed.

## DISCOVERY: metamon has a full team_construction backend

Path: `/workspace/metamon_ref/metamon/backend/team_construction/` on prod. Files:
`coordinate_ascent.py`, `restricted_game.py`, `model_fit.py`, `model_scoring.py`,
`matchup.py`, `simulation.py`, `cli.py`, `core.py`, `artifacts.py`,
`feature_baseline.py`, `feature_interaction.py`, `feature_sparse.py`.

**What it actually is**: meta-aware ELITE team generator, NOT a random sampler.

| Component | Function |
|---|---|
| `coordinate_ascent` | Swap mons one at a time to maximize objective |
| `coordinate_ascent_multi_start` | Run from N seeds → diversity of local optima |
| `restricted_game` + `zero_sum_equilibrium` | Build teams via Nash equilibrium |
| `strategy_pool_double_oracle` | Iteratively build pool approximating Nash |
| `model_fit` (baseline + interaction) | Learns matchup model WR(team_a, team_b) |
| `objective_vs_metagame` / `objective_vs_mixture` / `objective_vs_fixed_opponent` | Pluggable objectives |
| `sample_team` | Sample from constructed pool |

**Why this is HUGE** for our plateau direction: replaces both "manual scrape 100-150
teams" (Task #123) AND "hand-author template generator" with a single tool that:
- Generates teams with controllable diversity (multi-start with different seeds)
- Can target arbitrary objectives (beat smart bots / our SP snapshots / metamon Nash mixture / ...)
- Has infinite variety (continuous generation, not static pool)

**Strategic concern if used naively**: default objective likely optimizes against
metamon's metagame mixture → generated teams overfit to metamon's expected opponent
distribution = exactly the wrong direction (we'd memorize metamon-style teams).

**Smart use**: pass our own objective. Options ordered by safety:
1. `objective_vs_mixture(smart_bots + our_snapshots)` — teams that beat
   OUR opponents, not metamon's. Safe.
2. Multi-start coord ascent with NO objective optimization, just use the
   diverse local-optima outputs as a coverage set. Safest (no metamon-bias).
3. `objective_vs_our_model_weakness` — teams that exploit OUR specific
   weaknesses. Adversarial-curriculum-style. Powerful but risky (may hyperfit
   to whatever the matchup model thinks our weaknesses are).

**Setup cost estimate** (rough — needs validation): probably 5-7 days. Wire up:
- Build pokemon_pool artifact (probably just gen9ou usage data we already have)
- Fit baseline + interaction matchup models on a sample of our self-play data
  (or skip matchup model and use pure coord ascent on simulator-only objective)
- Wire `sample_team` into a TeambuilderAdapter we can pass to train_rl.py
- Generate N=500-1000 teams pre-training, use as MixedTeambuilder source
  (alongside procedural)

**Refined comparison table** (after this discovery):

| Approach | Pros | Cons | Setup cost |
|---|---|---|---|
| Manual scrape 100-150 | Real human teams, free | Overfits, manual | 3-6h |
| Hand templates | Controllable, infinite/template | Domain work; misses archetypes | 2-3 days |
| **Metamon team_construction (custom objective)** | **Meta-aware, infinite variety, tunable** | **Risky setup; may need matchup-model retraining** | **5-7 days** |
| Replay rehearsal during PPO | Direct hypothesis test, no team gen needed | Doesn't address team distribution per se | 1-2 days |

**Updated priority** (the one I endorse most strongly now):

1. **Replay rehearsal during PPO** — cheapest hypothesis test, ~1-2 days. If shows
   improvement → BC-decay hypothesis confirmed, team-distribution side is the
   missing piece.
2. **Metamon team_construction** (skip manual scrape and hand templates) — 5-7
   days but replaces both lower-tier approaches. Use coord ascent multi-start
   with custom objective (NOT metamon's default mixture).
3. **Combination** = replay rehearsal + metamon-generated teams. Probably the
   strongest team-side improvement we can make.

Manual scrape (Task #123) is now LOWER priority — keep around as fallback if
metamon team_construction setup proves too painful, but try the better tool first.

## User's questions / framing in this discussion

- "what really intersed me is the replay rehersal during ppo" — yes, that's the
  cheapest direct test of the hypothesis. Strongly recommend doing first.
- "metamon actually has a team generator in it" — confirmed and explored above.
  Much better than building from scratch.
- "with elite teams - absolutely wreak havoc" — generalist baseline + elite cap
  IS the goal; mixture (procedural + metamon-gen elite) addresses both halves.
- "mm x mm matrix is probably the least informative" — agreed; Minikazam likely
  wins those too. Deprioritized.
