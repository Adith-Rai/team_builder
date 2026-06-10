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

## S68 update (2026-06-05): refined hypothesis + active experiment

After Phase 3 v3 launch from snap_0139 (the 1178.4 Elo record holder), user
sharpened the AWR mechanism framing and pivoted the next experiment.

### Refined hypothesis chain (CONJECTURE markers throughout)

1. BC v10 trained on 1500+ Elo human gen9ou ✓ CONFIRMED (memmap pre-filtered)
2. BC = imitation → action distribution encodes elite patterns WITHOUT causal
   understanding ✓ true in principle (no policy gradient = no reasoning about
   why actions are good)
3. PPO + procedural-team training (random comps from 545-Pokemon usage pool)
   means model rarely sees synergistic-team contexts → never gets to execute
   elite plays for reward signal ✓ CONFIRMED training setup
4. Result: model becomes a "decent generalist" (the 70-74% smart_avg ceiling)
   but never refines elite-team execution. PARTIAL EVIDENCE — era4 ladder
   shows snap_0139 (1178) > BC (1135) on synergistic teams, so PPO doesn't
   actively REGRESS elite play. Better framing: PPO doesn't REFINE elite-team
   execution past BC's baseline. CONJECTURE
5. AWR provides steady stream of elite human-play state-action pairs during
   PPO → policy retains/amplifies these plays in sampled action distribution
   → when self-play (or eval) presents a synergistic team, model executes
   elite plays → PPO sees them succeed → reinforces with PPO's OWN refinement.
   The loop: AWR seeds → SP execution → PPO rewards → refines. CONJECTURE +
   MECHANISTICALLY GROUNDED

### Subtle correction

User initially framed (4) as "PPO forgets elite plays." Data doesn't fully
support active forgetting (snap_0139 > BC on synergistic-team ladder).
Refined framing: "PPO doesn't REFINE elite-team execution past BC's baseline"
— the model stays stuck at BC's untrained-elite-plays level rather than
improving them. AWR helps either way (retention if forgetting, amplification
if stagnation), so the experiment is still right.

### Phase 3 v3 = snap_0139 init data point (in flight, finishes ~04:30 UTC)

Tests "does AWR break the snap_0139 ceiling?" — strong V_θ gives clean
advantage discrimination but drifted policy means execution loop is weaker.
Will yield a useful data point regardless of outcome.

### BC-init AWR experiment (Run #3 + Run #4) — the real test

**Design**: two legs, identical except AWR. Difference = AWR's clean
contribution at BC + externals condition.

| Leg | Init | LR | Reward | Pool | Externals | AWR | iters |
|---|---|---|---|---|---|---|---|
| #3 (ablation) | BC v10 | 8e-5 | terminal | 151-entry prod | full (with minikazam) | NO | 200 |
| #4 (proposal) | BC v10 | 8e-5 | terminal | 151-entry prod | full (with minikazam) | mix=TBD (smoke), batch=16, binary | 200 |

**Adds minikazam** at temperature=0.01, weight=1.0, instances=3 — the
strongest MM (we lose ~91% vs it in prior data). Provides the strongest
external pressure for the refinement loop. NOT a "cheating" addition — it's
a legitimate strong opponent.

**Pre-launch smoke**: 5 iters at mix=0.10 AND mix=0.15 on prod (after Phase 3
v3 finishes) to calibrate AWR/total ratio at BC init. AWR loss magnitude may
differ from snap_0139 baseline (smoke 2.05% at mix=0.05). Pick mix landing
in 8-12% sweet spot.

**Cost**: ~$60 total, ~3 days wall.

**Operational sequence** (locked):
1. Now → ~04:30 UTC: Phase 3 v3 finishes on prod
2. ~05:00 UTC: smoke + Run #4 launch on prod
3. ~14:00 UTC: fishbowl_terminal finishes on dev → MM setup → Run #3 launch
4. ~+25hr: both complete, run eval_elo_ladder_cis_v2.py vs era4_chain_FINAL

**Decision tree**:
- Run #4 - Run #3 ≥ +10 Elo → AWR adds clean lift → promote to Phase 2 candidate
- Within ±5 Elo → AWR neutral → revisit hypothesis
- Run #4 < Run #3 by ≥10 → AWR actively hurts → diagnostic mode

**Full launch artifacts + checklists**: `memory/project_s68_bc_init_awr_design.md`

## S68 update (2026-06-07): Run #3 + #4 done; Run #5 + #6 firing to close 2×2

### Run #3 + Run #4 results (CONFIRMED, from evals.jsonl)

Both completed 200 iters. Eval registry data:

| Run | Pool | Peak smart_avg | At iter | End (iter 199) |
|---|---|---|---|---|
| Run #3 (BC + no AWR + proc-only) | 36 (dev) | **74.25** | iter 119 | 72.0 |
| Run #4 (BC + AWR + proc-only)    | 140 (prod) | **75.25** | iter 49 | (collapse → recover to 71.4 by iter 69) |

**Observation (PROVISIONAL)**: Run #3 sustained ~70-74 plateau throughout. Run #4 had higher PEAK (+1pp) but unstable — 75.25 → 68.8 in 10 iters then partial recovery. **AWR alone gave a small peak boost but introduced instability.** Direct comparison limited by pool-size + worker-count confound (Run #3 dev vs Run #4 prod) but the within-run trajectory shape is informative regardless.

### Refined hypothesis → next test (Run #5 + Run #6)

**The tug-of-war framing**: AWR pulls policy toward elite plays. PPO un-reinforces on procedural teams (Pokémon picks that don't synergize → elite plays don't pay off → PPO weights down). Net: AWR's pull is counter-productive without team contexts that support elite plays.

**Solution**: add synergistic (real ladder) teams to ~30% of training games. The PPO un-reinforce becomes PPO reward when elite plays land. AWR + syn teams = loop closes.

### 2×2 ablation (Run #5 + Run #6 firing 2026-06-07)

|              | Procedural-only        | + 30% syn               |
|--------------|------------------------|-------------------------|
| **No AWR**   | Run #3 done (pool=36)  | **Run #6 firing** (pool=36)  |
| **AWR**      | Run #4 done (pool=140) | **Run #5 firing** (pool=140) |

Same 60% hl_05_26 / 40% gl_05_26 syn mix for both, 200 iters each. Within-pod ablations (Run #5 vs #4, Run #6 vs #3) are clean apples-to-apples. Cross-pod (Run #5 vs Run #6) carries the same confound Run #3 vs Run #4 already had.

**Decision tree (when 2×2 lands)**:
- Run #6 - Run #3 ≥ +3pp smart_avg → syn teams help on their own → useful even without AWR
- Run #5 - Run #4 ≥ +3pp → syn teams + AWR closes the tug-of-war loop → confirms hypothesis
- Run #5 vs Run #6 (with pool-size caveat) → AWR's marginal contribution under syn condition
- All ~flat → either tug-of-war hypothesis is wrong, or 30% syn ratio is too low → revisit

**Infra shipped to support this** (commit `6997a2fa` mmap + `c8adf696` watchdog + `fb79af84` R2 bundles): see `memory/project_s68_team_data_arc_2026_06_07.md` for the architectural story. mmap-bundle scales O(1) per worker for multi-gen via kernel page cache; MM watchdog catches poke-env hangs in 30s instead of 300s (live-validated 3× in Run #5).

## S68 update (2026-06-09): Run #5 + #6 results → tug-of-war hypothesis REFRAMED, Run #7 firing

### Final smart_avg (200 iters)

| Run | Setup | Peak | At iter | Final iter 199 |
|---|---|---|---|---|
| Run #3 (done) | proc, no AWR | 74.25 | 119 | 72.0 |
| Run #4 (done) | proc, AWR | **75.25** | 49 | volatile (peak-then-collapse) |
| **Run #5 (done)** | syn, AWR | 72.75 | 189 | 70.0 |
| **Run #6 (done)** | syn, no AWR | **75.125** | 159 | 71.625 |

**All 4 runs landed in the 70-75 plateau.** Neither syn-team variant broke through. Run #6 nearly matched Run #4's all-time peak (75.125 vs 75.25) but then drifted back into the band.

### Per-opp WR slope over full 200 iters (externals)

```
Opp                  | Run #5 final WR / slope | Run #6 final WR / slope
mcts-fast            | 26.2% / +0.39           | 28.7% / +0.48
mcts-medium          | 30.0% / +0.49           | 30.4% / +0.54
mm-largerl           | 15.8% / +0.37           | 17.2% / +0.37
mm-mediumrl-aug      | 17.0% / +0.31           | 18.7% / +0.38
mm-minikazam         | 11.7% / +0.21           | 11.0% / +0.20
mm-syntheticrlv2     | 16.0% / +0.34           | 16.9% / +0.34
```

**Run #6 wins 5 of 6 externals end-of-run.** Run #5 only edges on mm-minikazam (the hardest opp). Both still climbing on external WR at iter 199 — neither plateaued in this signal yet.

### SP-pool WR — Run #6 dominates

Per-snap WR vs prior pool members (BC v10 + 150 historical anchors):

| SP Opponent | Run #5 final | Run #6 final | Lead |
|---|---|---|---|
| v10_padded (BC) | 62.7 | ~65.1 | Run #6 +2.4 |
| prod_1600g_snap_0009 | 56.7 | 64.5 | Run #6 +7.8 |
| snapshot_0049 (older) | 40.8 | 60.4 | **Run #6 +19.6** ⭐⭐ |

On older / BC-stylistic snaps, Run #6 wins decisively (+5-20pp). Closer on more recent snaps that themselves diverged from BC.

### bc_kl trajectories

Both runs used `--bc-anchor-coef 0.1`.
- **Run #5**: bc_kl stable at ~0.145 (AWR pulls toward BC plays → restrains divergence)
- **Run #6**: drifted upward 0.146 → 0.169-0.178 (without AWR's pull, anchor still bounds but lets it stretch)

Both stayed within anchor-allowed envelope. Neither collapsed. Run #6's drift CONFIRMS the anchor is the binding constraint.

### Tug-of-war hypothesis REFRAMED (PROVISIONAL)

**Original hypothesis (refuted as framed):** "AWR + syn closes the loop — AWR pulls toward elite plays, syn-team context lets PPO reward them."

**Data shows:** AWR + syn (Run #5) consistently UNDERPERFORMS no-AWR + syn (Run #6) across SP-pool, externals end-state, and smart_avg peak. AWR appears to be REDUNDANT with BC anchor — both pull toward BC.

**Reframed hypothesis (PROVISIONAL):** AWR and BC anchor are functionally two anchors. Together they over-constrain divergence. Removing either (Run #6 = no AWR; Run #7 = no anchor — see below) lets the policy explore more productively.

### Run #7 — testing the BC anchor hypothesis (firing 2026-06-09)

Launched 2026-06-09 16:11 UTC on prod. Same as Run #5 with `--bc-anchor-ckpt` and `--bc-anchor-coef` REMOVED entirely (`BCAnchor: OFF` verified). AWR still present (binary mix 0.15), syn 30%, full pool, MMs.

**Tests:** is the BC anchor the lid? If Run #7 escapes 70-75 plateau → anchor was the lid. If Run #7 also plateaus → BC v10 intrinsic ceiling, need BC v11.

**Safety**: AWR + syn teams + strong SP pool (snap_0139 etc) + MMs provide multiple grounding signals that Phase 1 v3 didn't have. Manual eyeball monitoring (no hard kill rules) for WR vs strong snaps + MM WR + smart_avg.

200 iters projected ~33hr. Currently at iter 6 (post-warmup, kl=0.018 — slightly elevated vs Run #5's ~0.010 due to no anchor restraint; not alarming).

### H2H eval queued

When snap upload completes, 4 snaps × 4 MMs × 500g tournament on metamon-competitive teams. Best snaps:
- Run #5 iter 189 (peak 72.75)
- Run #5 iter 199 (final 70.0)
- Run #6 iter 159 (peak 75.125)
- Run #6 iter 199 (final 71.625)

Compare to snap_0139 baseline (51% / 56% / 49% / 16% vs LargeRL/MedRL/SynRL/Mini).

The H2H result is the actual "is AWR worth it" answer — training-time WR underestimates H2H by ~30pp due to action sampling vs greedy argmax. AWR's contribution may show up here even if invisible in training-time signals.

### Bottom line so far (PROVISIONAL)

1. **Adding syn teams didn't break the plateau** — both Run #5 and Run #6 landed in the same 70-75 band as Run #3 + Run #4.
2. **AWR + BC anchor together appear redundant** — Run #6 (anchor only) explored more productively than Run #5 (anchor + AWR).
3. **The lid is either BC v10 (intrinsic ceiling) or BC anchor (configurable)** — Run #7 will distinguish.
4. **Per-opp slopes are still climbing** — runs haven't plateaued in external WR, only in smart_avg (which saturates at bot ceiling).
5. **H2H eval is the missing data point** — training-time WR systematically underestimates capability; the H2H result on MC teams may reveal AWR's true contribution.

## S68 update (2026-06-10 early UTC): Run #7 transition CONFIRMED + Elo ladder findings

After the Run #5/#6 H2H eval landed yesterday + a deep dive into Run #7 today, the picture has shifted meaningfully:

### Bot Elo ladder findings (CONFIRMED, n=30 g/matchup)

3-snap × 13-bot Bradley-Terry tournament: Run #5 peak iter189 (1184 Elo) + Run #6 peak iter159 (1160) + Run #7 snap_0039 (970). **Run #7 is ~200 Elo BELOW the anchored peaks** — real capability loss, not measurement artifact.

Best non-eval setup-spam punishers (15-30pp differential between policies):
- StrategicV2 (+30pp), SetupThenSweep (+24pp), HazardSense (+22pp), AntiSetupBot (+18pp), GreedySE (+18pp)

Original Strategic still beats my new StrategicV2 (991 vs 884) — loosening conditions made it worse.

Full results: `memory/project_s68_bot_elo_findings_2026_06_10.md` + `data/elo_ladder_*.json`.

### Run #7 transition CONFIRMED via iter 49 replay analysis

The "model is in flux, abandoning setup-spam" hypothesis (user's framing) is CONFIRMED by decision-pattern data:

- Early-setup (t1-5) in wins: **0.44 → 0.28** (-36% iter 39 → 49)
- Early-setup (t1-5) in losses: **0.55 → 0.31** (-44%)
- Setup % in wins: 14.48% → 12.29%
- Status targeting stable 60-61% (wasn't part of failure)
- Bot WR: 53% → 44% (exploration cost)

Replay-level evidence (eyeballed):
- iter 39 losses: setup-spam + type-blind + desperate Healing Wish
- iter 49 losses: principled pivots + direct damage + ZERO setup
- iter 49 wins: still Iron Moth sweep but with new glitches (SR spam when already up)

**Model is in "exploration valley"** — abandoning bad strategy, hasn't sharpened new one yet. Self-correcting via pool[-1] forcing + adaptive entropy + stronger pool members (snapshot_0149/0209) now BEATING current model.

### Updated Run #9 framing

Run #9 reframes from "fix the failure" to "accelerate in-progress shift":
- Heuristic-opp diversity provides direct gradient (15-30pp differential bots)
- Best pool candidates: StrategicV2, SetupThenSweep, HazardSense, AntiSetupBot
- Less urgent than I initially thought (natural mechanism is working)
- Decision tree: if iter 99 eval recovers toward 1100+ Elo, natural shift succeeded; if stays at 970, Run #9 needed; if drops <950, deeper problem

### Run #8 KILLED

Run #8 v2 (BC + no AWR + syn 50% on dev) was killed at ~iter 7 after the Elo data + iter 49 findings showed Run #7's evolution is more interesting than syn% ablation. Dev pod freed for measurement + future Run #9.

---

## S68 update (2026-06-09 late): H2H results + Run #8 firing + Run #9 in backlog

### H2H eval on metamon-competitive teams (n=500/cell)

4 snaps × 4 MMs = 16 matchups. Best snap: **run5_iter199 at 43.8% aggregate (+0.8 vs snap_0139 baseline)**. Modest net gain across runs; 3 of 4 snaps beat baseline aggregate. See `docs/S68_MM_EVAL_RESULTS.md` §4b for full matrix.

Headline patterns:
- All 4 RL snaps beat baseline on **Minikazam** (+1.8 to +3.6pp)
- Run #6 dominates LargeRL (+2.4 / +3.0pp)
- All snaps regress on MediumRL_Aug (-2 to -3.6pp)
- Run #5 vs Run #6 essentially tied in H2H despite Run #6 SP-pool dominance

### Run #8 (firing 2026-06-09 evening on dev pod)

**Setup**: Run #5 + `--syn-team-pct 0.50` (was 0.30). One-variable test of the syn lever. Everything else identical: BC anchor coef 0.1, AWR binary mix 0.15, lr 8e-5, full pool, 4 MMs × 3 instances.

Closes the syn% axis of the ablation matrix:
- Run #5 → Run #7 = pure anchor effect (currently firing)
- Run #5 → Run #8 = pure syn% effect

If Run #8 lands at smart_avg 70-75 plateau too → syn% NOT the lever; rules out "more elite contexts breaks the ceiling" framing. If significantly higher → syn IS a lever.

### Run #9 — opponent diversity (backlog, conditional on Run #7 result)

**Setup**: Run #7 + add 5/iter heuristic opponents (RandomPlayer + MaxBP + SimpleHeuristics) to PFSP pool. BC init + AWR + 30% syn + **no anchor** + heuristic-opp variety.

**Motivation (PROVISIONAL)**: Run #7's early per-opp data (iters 0-30) shows the model improving vs stochastic neural opps (SP-pool +5-14pp/10it, BC init +5.1pp/10it, mm-largerl +0.4pp/10it) but losing ground vs deterministic heuristics (smart_avg 65→60). One hypothesis: anchor was pulling toward "principled / heuristic-style play" which transferred to bots. Without anchor, model develops SP-specific tactics (baiting, prediction) that don't apply to deterministic opps.

User's framing: this is about **diversity in opponent distribution**, NOT smart_avg fix. Heuristics provide deterministic-style training signal that pure SP doesn't, regardless of the symptomatic smart_avg drop. The principled question: does opponent variety alone (no anchor) preserve robustness while keeping no-anchor's SP gains?

**REJECTED alternative**: anchor coef 0.05 (half-strength). User pushback: "we already found a pseudo-ceiling with it" — anchor at any coef > 0 just gives a different magnitude of the same lid. Either escape the lid or don't; half-doses just delay the answer.

**Cost**: ~$20, ~33 hr on dev pod. Cheap because heuristic bots run in-process (no MM server overhead).

**Conditional on**:
- Run #7 finishing as "specialization not collapse" — i.e., per-opp climbing while smart_avg drops
- Decision that ladder robustness matters enough to test (heuristic bots ARE proxies for "predictable opponents you should be able to dispatch")

**Will NOT pursue if**: Run #7 collapses (different problem, anchor was load-bearing) OR Run #7 lifts smart_avg back up on its own (no need for diversity intervention).

### Cross-references (for newer findings)

- `memory/project_s68_run5_run6_results_2026_06_09.md` — full results memo with evidence labels
- `memory/project_s68_team_data_arc_2026_06_07.md` — design + infra context
- `docs/MM_TRAINING_STRATEGIES.md` — what each MM we benchmark actually is
- `docs/S68_MM_EVAL_RESULTS.md` (with grain-of-salt section) — H2H baseline data
