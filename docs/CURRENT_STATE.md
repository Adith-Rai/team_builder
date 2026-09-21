# CURRENT STATE

**The single living source of truth for this project.** Everything else in
`docs/` is either a live reference (13 files at `docs/` root) or history
(`docs/history/`, journal entries true only as of their timestamp).

> **Convention**: this file is *maintained*, not appended. When something
> changes, edit the claim in place — do not add a dated section. Every claim
> carries an evidence label and a "verified" date. If you find a claim whose
> verified date is stale and you can't re-confirm it, mark it `UNVERIFIED`
> rather than deleting or trusting it.
>
> **Evidence labels**: `CONFIRMED` (measured) · `STRONG INFERENCE`
> (measurement + a standard assumption) · `CONJECTURE` (plausible, untested).

Last full review: **2026-09-20**

---

## 1. Where the project stands

**Goal**: elite Pokémon AI across all legal formats and all gens. Not just
gen 9 OU. Vertical-first sequencing (see `memory/project_strategic_frame.md`).

**Active arc**: complete encoding/tokenization audit → multi-gen + doubles/
triples *data* readiness → BC-phase performance. All local, zero cloud cost.

**Cloud is OFF.** RunPod abandoned after ~$3-4k total spend. Indefinite pause.
Do not propose pod work. *(CONFIRMED — user, 2026-09-20)*

**Parked, not resolved**: Run #9 relaunch, the Run #7 iter-99 H2H decision
gate, the `--resume` fix, Phase 2 launch.

**Last code commit**: 2026-06-13. A ~3 month gap precedes the 2026-09-20 session.

---

## 2. Infrastructure and data

### Compute
- **Local only**: RTX 3060 Laptop 6 GB / 16 GB RAM / Win10 / Python 3.11 /
  poke-env 0.10.0 / torch 2.2.1+cu121. Cannot carry a 90-worker collect loop.
- Disk: 277 GB free of 931 GB. Project directory is 134 GB.
  *(CONFIRMED 2026-09-20)*
- OneDrive is **not active** despite the path — no sync, not paid for.
  *(CONFIRMED — user, 2026-09-20)*

### R2 object storage — ALIVE
Bucket `team-builder-data`, account `86facaf13d3491bbc167027863b2f114`.
Credentials present locally (`[r2]` profile in `~/.aws/credentials`), AWS CLI
at `C:\Program Files\Amazon\AWSCLIV2\aws.exe`. **Zero egress fees.** ~$1.56/mo,
which is why it survived the pod teardown. Access recipe: `CLOUD_RUNBOOK.md`.

```
s3://team-builder-data/
├── datasets/human_v8_100k/   103.4 GiB, 23 objects — BC v10 training corpus (COMPLETE)
├── datasets/human_v8_5k/
├── models/                   BC + RL checkpoints
├── raw_data/
├── snaps_temp/
└── team_bundles/             .teampack mmap pool bundles
```
*(CONFIRMED 2026-09-20 — listed directly)*

**Not downloaded locally, deliberately.** The audit needs the schema, not the
bytes, and we will likely re-encode anyway.

### Local corpora
| Path | Contents | Status |
|---|---|---|
| `src/data/datasets/memmap_v8/` | 12,104 episodes / 360,881 records | 7.4 GB, intact |
| `src/data/datasets/human_v8_memmap/` | 159,934 episodes / 4,066,905 records | **metadata.json only** — the `.npy` files live in R2 |

### Replay data availability — the binding constraint on multi-gen
*(CONFIRMED 2026-09-20 via HuggingFace API)*

| Source | Coverage |
|---|---|
| `jakegrigsby/metamon-raw-replays` | 641,311 replays, **100% `gen9ou`** — single format. This is what `replay_to_memmap.py` streams |
| `jakegrigsby/metamon-parsed-replays` | **gen1/2/3/4 × {nu, ou, ubers, uu}** + `gen9ou` = 17 formats, plus `revealed_teams` and `replay_stats` |

**There is no gen 5, 6, 7 or 8 replay data anywhere in Metamon's collection.**
Obtaining it would require ingesting Showdown's own replay archive — a separate
problem, not currently planned.

### Static game data — multi-gen complete
`raw_data/`, prepared by the user pre-2025. Independent of the replay limitation.

| Path | Contents | Size |
|---|---|---|
| `raw_data/pokemon_usage/` | Smogon usage stats, **gen1-gen9**, 2014-11 → 2024-04 | 27 GB |
| `raw_data/pokemon/` | Pokedex (`1to8` + ScarletViolet_PBS) | 333 MB |
| `raw_data/movesets/` | moves.csv with `gen_added` | 9.6 MB |
| `raw_data/items/` | items.csv with `gen_added` | 144 KB |

**Consequence**: the encoding schema can be built for all gens even though
training is limited to gen 4 and gen 9.

---

## 3. Gen scope (decided 2026-09-20)

| Gens | Decision | Reason |
|---|---|---|
| 1-3 | **OUT** | Mechanics diverge too far — no abilities before gen 3, physical/special split is per-*type* before gen 4, gen 1 has its own damage formula. Nothing is lost taxonomically: every gen 1-3 species appears in gen 4+ dexes |
| **4** | **IN** | Damage formula, stat formula, phys/spec-per-move, abilities/natures/EVs/IVs all stable from here. Four tiers of replay data available |
| 5-8 | **LATER, not blocked** | No Metamon replay data. Obtainable from Showdown's own replay archive for similar formats — a separate ingestion job, deliberately deferred *(user, 2026-09-21)* |
| **9** | **IN** | Current primary format |

**Target format set**: `gen4{nu,ou,ubers,uu}` + `gen9ou` — five formats, two
gens, one mechanical regime. Gives real format-agnostic signal at no extra
mechanical cost.

`MULTIGEN_FEASIBILITY.md` recommends "gen 6+". **That recommendation is
refuted** — it reasoned purely about mechanics and never checked data
availability. Gen 6+ would yield gen 9 alone.

---

## 4. Capability — what we actually know

See `memory/project_s69_ceiling_evidence.md` for full detail.

### The absolute anchor (CONFIRMED, n=500/pair, metamon-competitive teams)
snap_0139 H2H vs frozen external models:

| Opponent | Our WR | smart_avg vs our bots |
|---|---|---|
| LargeRL | 51.0% | 70.2% |
| MediumRL_Aug | 56.2% | 66.7% |
| SyntheticRLV2 | 48.6% | 71.1% |
| **Minikazam** | **16.2%** | **91.9%** |
| *us* | — | *70-74%* |

**Minikazam scores 92% on the same metric we're stuck at 70-74% on.** The
metric has ~20 pp of headroom we are not taking. The plateau is not a
measurement artifact.

### Rate of progress (PROVISIONAL — monotonic across 3 points, each delta inside ±4.5 pp noise)
- `fishbowl_prod_lr1e-4_v1`, 150 iters: smart_avg 70-74%, mean 72%, **no climb**
- vs Minikazam across snap_0139 → 0249 → 0289: **18.4% → 20.4% → 22.2%**
- Reading: **~+3-4 pp per 150 iterations** against a ~30 pp gap.

### Two measurement rules, both learned the hard way
1. **Never compare Elo across ladders.** Bradley-Terry re-centers when the pool
   changes; frozen bots drift ~±50 Elo between ladders. Use frozen-opponent H2H
   win rate for capability claims. *(see `memory/feedback_elo_cross_ladder_invalid.md`)*
2. **smart_avg is DECOUPLED from MM win rate.** snap_0249 was the smart_avg peak
   *and* the worst of three snaps vs LargeRL/MediumRL_Aug. Use smart_avg only as
   a cheap in-run smoke signal, never as a capability claim.

Also: training-time per-opponent WR **understates H2H by ~30 pp** (sampling with
entropy vs greedy argmax at eval). Fine for direction, useless for level.

### Caveat on the reference set
Our MM pool is paper-era 2024 + Minikazam. Metamon's genuinely strong public
models — Kakuna 142M, Kadabra3, Alakazam, Abra — are **not** in our pool.
Beating LargeRL/MediumRL/SynthRL ≈ matching paper-era baselines, not "elite."
Minikazam is strong because it was distilled from Alakazam.

### Scale is NOT the variable *(STRONG INFERENCE, 2026-09-20)*

| Model | Params | Our WR vs it |
|---|---|---|
| SyntheticRLV2 | 200M | 48.6% |
| LargeRL | 195M | 51.0% |
| MediumRL_Aug | ~50M | 56.2% |
| **ours** | **~20M** | — |
| **Minikazam** | **small RNN** | **16.2%** |

We hold par against models 10x our size and lose 84-16 to one smaller than us.
Parameter count does not explain the gap.

What Minikazam has instead: **distilled from Alakazam** (a strong teacher),
`binary_rl.gin` wins-only filtering, parsed-replays v4, and **~5M self-play
battles**. Our runs are ~1600-2400 games/iter x 200 iters = **320-480k games** —
roughly 10x less self-play than Minikazam saw. Teacher quality and self-play
volume are the candidate levers, not scale.

### Representation is NOT the cap — REFUTED *(CONFIRMED 2026-09-20)*

Direct comparison of Metamon's observation space against ours (`metamon_ref/`
is cloned locally; `metamon/interface.py`).

Minikazam uses `PAC-OpponentMoveObservationSpace` (chain: `Default` →
`Expanded` → `TeamPreview` → `OpponentMove`; PAC re-introduces a tera bug for
backend compatibility) with tokenizer `DefaultObservationSpace-v1`. Their base
observation is **48 numerical dims for the whole battle state** plus a text
channel of ~106 whitespace-separated word tokens.

| | Metamon | Us |
|---|---|---|
| Categorical entity info (species/item/ability/move/type/status) | text tokens → embeddings | int ids → embeddings — **equivalent** |
| Engineered numerical features (precomputed type effectiveness, opp threat, switch defensive/offensive eff, 38 volatile bits, paradox, toxic fraction, combat state, 13-way boost one-hots) | **absent** | present |
| Stats | **base stats only** | computed stats |
| EVs / IVs / nature / substitute HP / duration counters | absent | absent |

**Minikazam beats us 84-16 while seeing strictly less structured information
than our model does.** "The model cannot perceive enough" therefore cannot
explain the ceiling.

Note: their "text embeddings" are mechanically identical to our embedding
tables — a fixed word count serialized as a string, each word mapped to an id
and a learned vector. On raw entity information we are roughly equal; we are
richer only in *derived/engineered* numerical features.

**Consequence**: the encoding/tokenization work is justified as **multi-gen
groundwork and completeness (including the future team builder)** — NOT as a
plateau fix. Do not expect it to raise the ceiling, and do not read its result
as evidence either way about the ceiling.

### Is our observation space too large? *(CONJECTURE — efficiency question, not a ceiling explanation)*

Plausible mechanism: a 20M model spends a large share of capacity compressing
285 dims x 12 Pokémon plus 109 x 4 move slots into a per-timestep summary.
Metamon deliberately keeps per-step thin (3 layers at d=100 over ~10 thin
tokens) so the sequence model does the strategic work; ours is 6 layers at
d=256 over 16+ rich tokens.

**But it cannot be the ceiling explanation**: if a rich observation were
crippling us we would lose to everyone, and we are at par with 200M models
using this exact observation space.

Settling test if it ever matters: BC-level ablation, matched parameter count,
our full obs vs a Metamon-like reduced obs. Local, no cloud cost.

**Capacity split — already addressed.** `memory/feedback_capacity_allocation.md`
(Session 33) describes a ~1:1 spatial:temporal ratio vs Metamon's 5-8x
temporal-heavy. That reallocation has since been done. Current code:
`d_model=256 / n_spatial_layers=6` and `d_temporal=512 / n_temporal_layers=4`,
with an explicit comment citing the "shift capacity from spatial to temporal"
recommendation. **The S33 memo is stale on this point**; S66's "temporal-heavy,
64% of params" is the accurate current description.

---

## 5. Codebase audit (2026-09-20)

Full sweep of the encoding/data layer. All findings verified against live code.

### The root cause: `FormatConfig` exists and is never threaded

`format_config.py` is a genuinely good abstraction — a frozen dataclass with
`gen`, `team_size`, `n_active`, `n_bench`, `n_moves`, `n_switches`,
`n_actions`, validation asserts, and commented-out doubles/triples configs.
Its header states the intent: *"Every magic number that changes between
singles/doubles/triples lives here."*

**But every consumer imports the `FORMAT_SINGLES` singleton and binds
constants at import time:**

```
features.py:38-41          N_TYPES, MAX_BENCH, MAX_MOVES, N_STATS
model_transformer.py:282   N_TOKENS = total_tokens(FORMAT_SINGLES)
model_transformer.py:309   return FORMAT_SINGLES
rewards.py:76-77           ts = FORMAT_SINGLES.team_size
```

`format_from_str()` — which parses gen 1-9 out of a format string — **has zero
callers.** Dead code.

This single threading failure is the root of the gen dimension, most hardcoded
constants, and the team-layout blocker for doubles. **Fixing it addresses all
three in one pass.** The right shape already exists; it just isn't wired.

### Already correct — do not rebuild these

- **Type chart is gen-aware for free.** `_compute_type_effectiveness`,
  `_opp_type_threat` and both switch-effectiveness helpers delegate to
  poke-env's `Pokemon.damage_multiplier()` — *"no hardcoded type chart."* Gen
  4's missing Fairy and Steel's extra Dark/Ghost resistances come free,
  provided the Battle is built with the right gen. (This also eliminated an old
  bug class: `history/STATUS.md:2439` records hardcoded type dicts missing
  NORMAL → GHOST immunity.)
- **Vocabulary is already multi-gen.** `vocab.py` builds the species/move/
  ability ID space from poke-env's `GenData.from_gen(gen)` for **gen 1-9**,
  unified into one ID space — sourced from poke-env, not hardcoded tables.
- **Move target semantics are already encoded.** `_TARGET_MAP` yields a 6-way
  one-hot over `SELF / NORMAL / ALL_ADJACENT / ALL_ADJACENT_FOES / FOE_SIDE /
  ALLY_SIDE`, plus `_SELF_TARGETS` for boost routing. Spread-move semantics —
  the thing that mechanically distinguishes doubles — is in the move features
  today.
- **Team generation is gen-parameterized**: `get_ban_list(gen=9)`,
  `_default_tiers(gen=9)` — defaulted to 9, but threaded.
- **Opponent-side leakage: none found.** Opponent abilities gated on
  revelation; opponents get `base_stats` while own side gets computed `stats`;
  explicit `ability_known` / `item_known` visibility flags. Verified across
  `_encode_pokemon`, `_encode_field`, `_encode_team`.
- **Eval-team leakage guard holds.** `MetamonCompetitiveTeambuilder` appears
  only in eval paths. `train_bc.py`'s use is inside `eval_vs_bots()`.
  `train_rl.py` never imports it — training uses `ProceduralTeambuilder`.

### New encoding gaps found (NOT in the May memo)

The May catalogue covered nature/EVs/IVs/substitute-HP/duration-counters.
These are additional, all player-visible, all currently discarded:

| Gap | Why it matters |
|---|---|
| **Team preview is dropped after the first reveal** | `battle.opponent_team` returns the team-preview roster only while `_opponent_team` is empty. From turn 1 on, the model sees only revealed mons — a human retains all 6 for the whole battle. **Information loss, not leakage.** Metamon built `TeamPreviewObservationSpace` for exactly this. Gen 9 only; gen 4 has no team preview |
| **`possible_abilities` discarded for opponents** | The fallback only fires for our own side. A player knows the legal ability set of a revealed species (reveal Landorus-T → ability is Intimidate). Encoding it as a multi-hot is player-visible and leak-free |
| `base_species` | Forme tracking — Landorus vs Landorus-T, Urshifu/Rotom forms. Metamon carries this |
| `current_hp` / `max_hp` | Exact HP for own side; we encode only `hp_pct`. Confirms the May memo's Tier 2 item |
| `preparing_move` / `preparing_target` | We know a mon *is* charging, not *what*. Same pattern as "choice lock — which move" |
| `previous_move` | Explicit last move per Pokémon; currently only inferable via the temporal stack |
| `gender` | Attract, Rivalry, Cute Charm |

Already covered (verified, do not re-add): `first_turn`, `must_recharge`,
`preparing`, `protect_counter`, `status_counter` — all in `_combat_state`.
Correctly ignored: `pokeball`, `shiny`, `stab_multiplier`. Parked (no data for
those gens): `available_z_moves` (gen 7), `is_dynamaxed` (gen 8).

### Two architectures coexist — BC defaults to the legacy one

| | Module | Class | Status |
|---|---|---|---|
| Legacy | `model.py` (header: `policy_heads_v8.py`) | `PokeTransformer` | **`train_bc.py` default** (line 470) |
| Current | `model_transformer.py` | `TransformerBattlePolicy` | **behind an opt-in flag** (line 396) |

`arch_compat.py` bridges them by duck-typed dispatch: the legacy model exposes
`forward_spatial / action_encoder / policy_head / value_head` separately so
CIS and `ppo_update` can share one spatial pass; `TransformerBattlePolicy` has
a monolithic `forward(batch, history)`.

**Anyone running `train_bc.py` without the flag trains the legacy
architecture.** Carrying both through an encoding-schema rewrite means doing
the schema work twice. **Recommendation: retire the legacy arch as part of the
rewrite**, after confirming nothing still depends on legacy checkpoints.

### Dead paths and local blockers

- **Dead**: `mp_collect_v2.py` (only self-imports), `mp_collect_v3.py` (zero
  importers), `format_from_str()`. **Not dead**: `mp_disk_collect.py` —
  `train_rl.py` imports it as an alternate path.
- **19 hardcoded `/workspace/` paths across 12 files** — mostly
  `--procedural-teams-path` defaulting to `/workspace/raw_data/pokemon_usage/
  2024-04`. That path does not exist locally, so those eval tools need explicit
  overrides. `train_rl.py:1454` and `h2h_diag.py:66` do it correctly via
  `__file__`.
- Hardcoded team size where the config exists: `features.py:604,609,648`
  (`/6.0`, `6 - opp_fainted`), `_encode_action_slots` (`np.zeros(9)`,
  `range(4)`, `range(5)`), `collate_seq` tensor shapes
  (`torch.zeros(B,T,6,…)`).
- `n_types: int = 19` is commented *"constant across all formats/gens"* —
  false for gens 1-5. Functionally harmless (superset multi-hot), but the
  comment misleads.

### BC pipeline — candidates pending a profile

| Setting | Current | Note |
|---|---|---|
| `--workers` | **0** | Dataloading in the main process. `persistent_workers=(nw>0)` is conditional but off by default |
| `--batch-size` | **16** | Small |
| Precision | **fp16 + GradScaler** | RL side moved to bf16 (no scaler needed, more stable) |
| `GradScaler` call | `torch.cuda.amp.GradScaler()` (line 557) | Deprecated API; line 86's signature already uses modern `torch.amp` |

`collate_seq` pre-allocates then fills per-sample — structurally the pattern
that produced the RL-side `aten::cat`/`copy_` cost.

**None of these are claimed as wins.** Every one resembles an "obviously
faster" change the S59-S68 arc refuted, and **S60 Fix #3 specifically refuted
vectorizing the RL `collate_episodes`** — different function, different scale,
so it neither confirms nor rules out the BC case. Profile first. The workload
is genuinely different (memmap IO vs orchestration), so expect different
answers than the RL arc gave.

### Checkpoints — the phase3 lineage is gone

260 `.pt` files survive locally (Phase 1 v3, rl_v9/v10 ppo_phase1, phase2
stage1/lr3e5/diversity/vf05, bc v8). R2 holds `models/bc/v10_cloud_gen9/`
(**the BC v10 base — the one that matters**), `models/rl_v10/ppo_phase1_v2_cloud/`
and `ppo_phase1_v3_cloud/`, and `snaps_temp/run7_snap_0039.pt`.

**Missing from both: Run #5 peak (iter 189), Run #6 peak (iter 159), and Run #7
past iter 39** — the strongest models we ever measured (Elo 1184 / 1160). Only
launch scripts, logs and result JSONs survived. Eval artifacts in
`data/eval_artifacts/s68/` are intact, which is why `S68_MM_EVAL_RESULTS.md`
remains trustworthy even though it can't be reproduced.

> ⚠️ **Do not cancel R2.** At ~$1.56/mo it holds the only copy of the 103 GB
> BC corpus and the BC v10 base checkpoint. Nothing else has them.

---

## 6. Open — genuinely unresolved

| Question | Status |
|---|---|
| ~~Run #7 (BC anchor removed): exploration valley or real degradation?~~ | **CLOSED AS UNRECOVERABLE 2026-09-20.** The iter-99 gate never ran, the resume failed mechanically, and **every checkpoint past iter 39 is gone** (§5). Only re-running the experiment answers it — that is cloud work |
| Run #9 (heuristic-opp diversity) | **Never ran.** Designed, coded, launch script written; hung on dev because heuristic bots raised exceptions inside CIS. Root-caused and fixed 2026-06-12/13. **Untested, not refuted** |
| `--resume` mechanism | Two of three issues appear fixed (`d1c15295`, `05a04d54`). The third — PFSP win_rates being path-keyed so forked dirs start blind — is **unverified against current code** |
| ~~Is representation the cap?~~ | **ANSWERED 2026-09-20 — NO.** See §4. Metamon's obs is strictly poorer and Minikazam still beats us 84-16 |
| Why does Minikazam win, then? | Open. Candidates, in rough order of support: distillation from a strong teacher (Alakazam), ~10x more self-play volume, offline RL on human replays vs our online PPO, opponent-pool composition. None tested |
| Did the type chart / tokenizer produce the historical jumps? | Unverified. Predates all reviewed material |

---

## 7. Settled — do not reopen

- AWR and the BC anchor are **functionally redundant** — both pull toward BC.
  Run #6 (no AWR) beat Run #5 (AWR). AWR measured ~24 Elo of micro-improvement,
  no decision-pattern effect. *(CONFIRMED)*
- The full infra REFUTED list lives in `REFUTED_LOG.md` and
  `memory/project_cloud_rl_arc_parked.md` — `--compile` at prod scale, MPS,
  vectorized collate, "more workers helps", pure-SP, `empty_cache`, Liger,
  the simulator rewrite, freeze-spatial as a permanent measure.
- Never mix the 16 metamon-competitive eval teams into training (leakage).

---

## 8. Known traps

- **Two `data/` directories.** The real corpus is under
  `pokemon-ai-starter/pokemon-ai/src/data/`, *not* `pokemon-ai/data/`. Easy to
  conclude the corpus is missing when it isn't.
- **`LOOKUP_SCHEMA_VERSION = 4`** and unchanged since May 2026, so the encoding
  gap catalogue in `memory/project_encoding_audit_phase2_todo.md` is still a
  valid baseline. Bump it whenever the schema changes.
- `docs/history/` entries are journal snapshots. Several contain framings later
  refuted. Trust this file over them.

---

## 9. Live docs (everything else is in `history/`)

| File | Purpose |
|---|---|
| `CURRENT_STATE.md` | **this file** — the only doc that claims to be true *now* |
| `ARCHITECTURE.md`, `CURRENT_ARCH_PATHWAY.md` | architecture + canonical data flow |
| `ARCHITECTURE_REVIEW_2026_06_07.md` | open TODO: switch-action representation ablation |
| `MULTIGEN_FEASIBILITY.md` | multi-gen scoping — **its "gen 6+" recommendation is refuted, see §3** |
| `REFUTED_LOG.md` | maintained don't-retry list — **infrastructure** side |
| `TRAINING_LEDGER.md` | maintained tried/outcome ledger — **training** side |
| `PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md` | plateau framework + experiment queue |
| `S68_MM_EVAL_RESULTS.md` | the MM anchor measurements |
| `MM_TRAINING_STRATEGIES.md`, `METAMON_LEARNINGS.md` | how the reference models were trained |
| `MODEL_REGISTRY.md` | checkpoint registry |
| `CLOUD_RUNBOOK.md` | R2 access (still needed), pod setup (parked) |
| `COMMANDS.md` | command reference |
| `SESSION_BOOT_PROTOCOL.md` | standing orders (mirrors memory) |
