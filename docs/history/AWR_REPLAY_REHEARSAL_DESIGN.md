# AWR Replay Rehearsal — Design Memo (S68, Task #125 Phase 1)

Owner: claude · Date: 2026-06-03 · Status: **DRAFT — awaiting user signoff**

Companion to:
- `docs/REPLAY_REHEARSAL_AWR_VS_OFFPOLICY_PPO.md` (AWR vs Off-Policy PPO theory)
- `docs/PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md` (strategic context)

This memo documents what I learned reading the code, the open questions I've
resolved (and how), the remaining open questions for you, and the proposed
implementation plan. **No code touched yet.** Awaiting your signoff before
branch + implementation.

---

## 1. Goal recap

Mix small fraction (5-10%) of OFFLINE replay-based AWR loss into each PPO iter.
Tests the plateau hypothesis: BC-learned elite patterns fade because PPO never
gives them reward signal; replay rehearsal keeps them in scope by reinforcing
elite actions weighted by their outcome.

**First-try recipe**: AWR (no behavior policy needed). If validates direction but
plateaus → escalate to Off-Policy PPO.

**Validation success criterion**: AWR variant beats baseline by ≥2pp on MM-tier WR
(vs LargeRL on metamon-competitive teams) over a 30-50 iter comparison. Smart_avg
likely won't move (bot-anchored, already saturated).

---

## 2. Key findings from code read

### 2.1 Loss assembly is clean — BC anchor is a perfect template

`ppo.py:594-709` (`_ppo_loss_batched_internal`) shows the existing BC anchor
pattern:

```python
if bc_logits is not None:                              # bc enabled
    bc_lp = F.log_softmax(bc_logits.float(), dim=-1)
    bc_p = F.softmax(bc_logits.float(), dim=-1)
    bc_kl_per_pos = (bc_p * (bc_lp - lp)).sum(-1)      # (B, L_max)
    bc_kl = (bc_kl_per_pos * pad_mask_f).sum() / n_valid
    total_loss = total_loss + bc_anchor_coef * bc_kl
else:
    bc_kl = torch.zeros((), device=device)
```

AWR slots in identically — same pattern, different math:

```python
if replay_batch is not None:                           # awr enabled
    replay_logits = forward_replay(model, replay_batch)
    replay_lp = F.log_softmax(replay_logits.float(), dim=-1)
    replay_chosen_lp = replay_lp.gather(-1, replay_actions.unsqueeze(-1)).squeeze(-1)
    # advantage from V_θ on replay states
    replay_V = extract_mean_from_v_logits(forward_replay_v(model, replay_batch))
    advantage = replay_R - replay_V
    weight = torch.exp(advantage / beta).clamp(max=clip_high)
    awr_loss_per_pos = -weight * replay_chosen_lp
    awr_loss = awr_loss_per_pos.mean()
    total_loss = total_loss + awr_mix_weight * awr_loss
```

Two loss paths exist (padded `_ppo_loss_batched_internal` and packed
`_ppo_loss_packed_internal` at `ppo.py:725-`). **Decision deferred**: do AWR
on the EAGER batched path first; both loss paths are pure-python so adding
AWR to both is trivial later if needed.

### 2.2 PPO call site is clear

`train_rl.py:1643-1657`:

```python
loss_info = ppo_update_batched(
    model, optimizer, episodes, device, cfg,
    ...
    bc_ref=bc_ref,
    bc_anchor_coef=args.bc_anchor_coef,
    ...
)
```

AWR adds: `awr_loader=awr_loader, awr_mix_weight=..., awr_beta=..., awr_binary=...`

### 2.3 BC anchor loading pattern reusable

`train_rl.py:1412-1425`: load v10 checkpoint with `load_checkpoint`, set
eval + freeze, pass as `bc_ref`. AWR doesn't strictly need a reference model
(advantage = R − V_θ on the REPLAY state, computed by the live model), but if
we ever do off-policy PPO, we could reuse this pattern as the π_behavior
estimator. Out of scope for AWR.

### 2.4 Replay data: BC v10 memmap is reusable

Confirmed obs encoding compatibility between BC v10 training data and current
features.py:

| Dim | BC v10 memmap (`memmap_v8/metadata.json`) | Current `features.py` | Status |
|---|---|---|---|
| POKEMON_CONT | 285 | 285 (line 409) | ✓ MATCH |
| FIELD_CONT | 52 | 52 (line 653) | ✓ MATCH |
| TRANSITION_CONT | 51 | 51 (line 957) | ✓ MATCH |
| MOVE_SLOT_CONT | 107 | 109 (line 1266) | ⚠ +2 (zero-pad — `dataset.py:62-78` handles) |
| SWITCH_SLOT_CONT | 28 | 30 (line 1399) | ⚠ +2 (zero-pad — same) |

**This is the big simplifying find**: no replay re-encoding needed.
`MemmapDataset` already zero-pads automatically with a warn for the move/switch
dims. The strict-equality dims (poke/field/trans) all match. AWR can reuse the
exact same data loader BC v10 used.

### 2.5 Memmap structure already has what we need

`dataset.py:84-103` enumerates the memmap fields. Notably:
- `action[N]` — what was done
- `result[N]` — terminal outcome (±1 win/loss/0 tie)
- All per-state encoded fields (our_pokemon_*, opp_pokemon_*, field_*, move_*, switch_*, legal)

**`result` is terminal-only.** No per-step KO/HP/immune. So AWR with the existing
memmap gets BINARY reward only. This is fine — matches metamon's `binary_rl.gin`
(SyntheticRLV2 was trained this way). If we want shaped reward later, need to
re-parse raw replays through current `compute_reward()` — separate, larger
effort.

### 2.6 Data availability locally

- `data/datasets/memmap_v8/` — 360K records / 12K episodes, ALREADY extracted ✓
- `data/datasets/human_v8_memmap/` — 4M records / 160K episodes, .npy MISSING.
  990MB tar.gz exists at `data/datasets/human_v8_memmap.tar.gz`. Need to extract
  for production-scale AWR; the small memmap is enough for smoke validation.
- On cloud pod: per `reference_cloud_pods_usage.md`, large memmap should already
  exist there (BC v10 was trained on the cloud).

### 2.7 Value head outputs categorical (two-hot)

`ppo.py:670-674`: value is categorical over V_bins, target is two-hot encoded
from scalar return. For AWR we need scalar V_θ(s) for the advantage. Extract
via softmax over v_logits × bin_centers (the same way Tier 3 already does
implicitly). Need to verify model has a helper for "v_logits → scalar"; if not,
add one.

### 2.8 Compile boundary considerations — deferred

Per `project_bc_anchor_design.md` + `project_s60_fix2_design.md` + S60 Fix #2
codebase comments: BC anchor through the compile boundary took multiple
sessions to get right (closure-bound `bc_anchor_enabled`, 0-dim tensor coef
to avoid recompile, per-chunk KL trip handling). Repeating that pain for AWR
is unnecessary v1 work.

Per S62 finding (`project_s62_fix2_prod_validation.md`): `--compile` is NOT in
the canonical Phase 2 stack anymore (8% slower at prod scale). **So AWR on
EAGER path is the right v1 path** — no compile boundary work, no recompile
risk, no closure-flag plumbing.

---

## 3. Resolved open questions (with rationale)

| Q | Decision | Why |
|---|---|---|
| Obs space match? | YES — reuse BC v10 memmap directly | All strict-equality dims match; move/switch zero-pad via existing dataset.py mechanism |
| Reward source? | Binary terminal (result field from memmap) | Matches `binary_rl.gin` AWR variant (SyntheticRLV2); shaped reward requires raw replay re-parse (separate workstream) |
| AWR loss in eager or compile? | Eager only in v1 | Compile not in canonical stack; compile-boundary pain documented in BC anchor sessions; AWR v1 should validate hypothesis cheaply first |
| Mix granularity (per-chunk vs separate pass)? | **Separate AWR pass per iter** (recommended) | Cleaner separation: PPO chunk loop unchanged; AWR is its own forward+backward+step inside ppo_update_batched after PPO loop. Less coupling, easier to A/B |
| Starting iter | Resume from snap_0139 (lr8e-5 record) | Same baseline as fishbowl_prod for clean A/B comparison. Don't start from BC v10 — that mixes "AWR helps PPO" with "AWR helps BC". |
| Replay sampler scope | Pre-sampled batch per iter, N=256-512 transitions | Small enough for cheap forward, big enough for stable gradient. Tune later. |
| Replay shuffling | Per-iter random sample from full memmap | Standard offline RL pattern |
| AWR variant first try | **Binary AWR** (1[A>0] × log π) | Simpler, no β tuning, matches `binary_rl.gin`. exp(A/β) variant requires β sweep. |

---

## 4. Open questions for YOU (need decision before code)

### Q1: Which run to A/B against?

**Option A**: snap_0139 baseline, fresh 30-50 iter run with AWR vs without on
dev pod. Clean comparison, controls for "did externals + AWR help vs neither."

**Option B**: snap_0289 (= fishbowl_prod end). Continue from "externals already
in pool" state, layer AWR on top. Tests whether AWR adds beyond externals.

**Recommendation**: Option A first. Cleaner causal story.

### Q2: Mix weight starting value? — DECIDED + ADAPTIVE

**Decision: start 0.05 for the SMOKE, calibrate validation-run mix weight
from observed smoke data.** Don't pick the validation weight a priori.

**Why we can't pick it a priori**: AWR loss is CE-scale (~1-2), BC anchor
loss is KL-scale (~0.01-0.05). So 0.05 × AWR_loss could easily be 10-50×
larger gradient contribution than 0.10 × BC_KL (which we know works without
destabilizing). The right mix weight depends on the relative magnitude that
emerges from real data; impossible to predict from the mix weight alone.

**Calibration rule** (apply at 5-iter smoke):

| AWR loss contributes... | Validation mix weight |
|---|---|
| <2% of total loss | **bump to 0.10** (current weight is measurement noise, not intervention) |
| 2-10% | **stick with 0.05** |
| >10% AND KL stable (avg_kl < target_kl × 3) | **stick with 0.05** (good signal, working as intended) |
| >10% AND KL inflating | **drop to 0.02-0.03** (AWR is overpowering PPO, need to back off) |
| `awr_weight_max` >> 1.0 frequently | Lower `--awr-clip-high` or confirm binary mode |

**Bounds reasoning**:
- <0.02 is below noise floor — counter-signal too weak to reverse the
  hypothesized BC-pattern fading.
- >0.10 risks "AWR becomes the main signal" → converges back toward BC v10,
  defeating the purpose. Current training already navigates reward-hack
  territory (bc_kl > 0.20 in fishbowl_v2_resume per memory) — don't add a
  second strong attractor.

This is why we DO BOTH smoke gates (Q4) — the 5-iter is not just a no-crash
check but a calibration probe.

### Q3: Should AWR replays be FILTERED to 1500+ Elo only?

The full memmap has all ratings. BC v10 was trained on filtered 1500+ replays.
For AWR we want HIGH-quality demonstrations — sampling random replays would
dilute signal.

**Recommendation: filter at memmap-load time to 1500+ Elo episodes only**
(metadata in the memmap — need to verify `episode_index.npy` has a rating
column; if not, this is implementation work).

### Q4: Smoke validation gate before validation run?

Two options:
- Smoke = 1 iter (~7 min) that AWR loss runs without crash, logs sane numbers
- Smoke = 5 iter mini-run that AWR doesn't catastrophically destabilize training

**Recommendation: both — 1-iter smoke first, then 5-iter stability smoke before
the 30-iter validation run.** Cost: ~50 min total. Cheap insurance.

### Q5: Where do we run this?

Dev pod is busy with era4_v3 ladder (expected ~5-6h total wall, ~3h in). Prod
pod has the two-snap MM eval in flight (~40 more min). **Recommendation: wait
for both to complete, then run AWR validation on prod pod** (prod has the full
human_v8 memmap extracted, dev would need to re-extract).

---

## 5. Implementation plan (phased)

### Phase 2A — Branch + skeleton (~1 session, no functional changes)

- Branch off master: `feat/replay-rehearsal-awr`
- Add CLI flags to `train_rl.py`:
  - `--awr-replay-memmap PATH` (default None — disabled)
  - `--awr-mix-weight FLOAT` (default 0.05)
  - `--awr-batch-size INT` (default 512)
  - `--awr-min-rating INT` (default 1500)
  - `--awr-binary` (default True for v1)
  - `--awr-beta FLOAT` (default 1.0, used only if not --awr-binary)
- Wire flag plumbing, no behavior change yet. Add WARN if memmap provided but
  flag combo is incoherent.

### Phase 2B — AWR replay loader (~1 session)

- New module: `awr_replay.py`
- `AWRReplayBuffer(memmap_path, min_rating)` — wraps `MemmapDataset`, filters
  by rating, exposes `.sample(batch_size, device)` returning a collated batch
  in the same format `forward_ppo_sequence` expects (one episode per slot, or
  flat transitions — TBD based on the forward shape needs).
- Smoke: standalone test that sampling works + shapes match what
  `forward_ppo_sequence` consumes.

### Phase 2C — AWR loss + integration (~1 session)

- Add `_awr_loss(model, replay_batch, beta, binary, clip_high)` in `ppo.py`
- Modify `ppo_update_batched` to optionally do an AWR step AFTER the PPO loop
  (or after each PPO epoch — TBD based on stability)
- Log new stats: `awr_loss`, `awr_advantage_mean`, `awr_weight_max`,
  `awr_pos_advantage_frac`
- Smoke: 1-iter run on prod pod, verify AWR loss runs, gradients flow, no NaN

### Phase 2D — Stability + mix-weight calibration smoke (~1 session)

- 5-iter mini-run on prod pod from snap_0139 with `--awr-mix-weight 0.05`
- **Stability gates**: no NaN, avg_kl within target_kl × 3, smart_avg within
  -3pp of baseline (small drop OK at iter 5, big drop = destabilization)
- **Calibration data** (logged per-epoch):
  - `awr_loss / total_loss` fraction → applies Q2 calibration table to
    pick validation mix weight
  - `awr_weight_max` — extreme weights flag β/clip_high tuning
  - `awr_pos_advantage_frac` — should hover ~0.5 if model and data are aligned;
    extreme (≪0.1 or ≫0.9) means our V_θ is mis-calibrated to the replay
    distribution
  - `awr_grad_norm / ppo_grad_norm` (if `--diag-grad-norms` on) — confirms loss
    fraction translates to gradient fraction

### Phase 3 — Validation run (~1 session monitoring + 30-50 iter wall ≈ 4-6h)

- 30-50 iter run on prod pod, snap_0139 baseline
- A/B: with vs without `--awr-replay-memmap`
- Compare: smart_avg trajectory + MM-vs-our_model on LargeRL (cheapest MM eval)
- Decision gate: ≥2pp MM-tier WR gain → escalate to longer run + more MMs
  + Off-Policy PPO consideration. <1pp gain → AWR doesn't help here,
  reconsider hypothesis.

### Out of scope for this arc (deferred)

- Shaped reward on replays (requires raw replay re-parse, separate workstream)
- Compiled path AWR (`--compile` not in canonical stack)
- Packed-mode AWR (deferred until packed-mode is canonical AND eager AWR validated)
- Off-Policy PPO (only if AWR validates direction but plateaus)
- Multi-step credit assignment (GAE on replays) — needs the shaped-reward arc first
- Custom replay weighting (per-rating bucket, per-team-archetype, etc.)

---

## 6. Smoke + validation plan summary

| Stage | What | Cost | Pass gate |
|---|---|---|---|
| 2A done | Skeleton commits + flag plumbing | 0 (just dispatch) | Existing tests pass |
| 2B done | Replay loader unit-test | ~$0 (local) | Shape match, sample diversity |
| 2C smoke | 1 iter on prod | ~$0.25 | No crash, AWR loss logged |
| 2D smoke | 5 iter on prod | ~$1.25 | No destabilization vs eager baseline (within noise) + calibration data for Q2 table |
| Phase 3 A/B | 30 iter × 2 runs on prod | ~$15 | MM-tier WR Δ ≥ 2pp |

Total budget if all phases run: **~$17 + 4-5 sessions**.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| AWR destabilizes PPO | Medium | Small mix weight (0.05), eager-only first, 5-iter stability smoke |
| Memmap data has rating metadata missing | Medium | Phase 2B verify; if missing, mod replay_parser to add OR skip rating filter v1 |
| Binary AWR signal too weak (only terminal reward) | Medium | This is the FIRST-TRY recipe; if weak, escalate to shaped reward (separate arc) OR exp(A/β) variant |
| Conflict with bc_ref / bc_anchor still active | Low | AWR and bc_anchor can both run — they're orthogonal (one anchors to v10's distribution, other reinforces high-A replay actions). May want to ablate bc_anchor off for clean AWR A/B though. |
| AWR gradient dominates PPO (mix weight too high) | Low | Logged stats (`awr_loss` magnitude) make this visible; tune down |
| Pod time conflicts | Medium | Wait for era4_v3 + two-snap MM evals first; Q5 above |

---

## 8. Decisions locked (S68 wrap, 2026-06-03)

| Q | Decision | Source |
|---|---|---|
| Q1 baseline | snap_0139 (clean) | User: "go with your recommendations" |
| Q2 mix weight | 0.05 for smoke, ADAPTIVE for validation per §4 Q2 table | User: "will 0.05 be too weak?" → updated to adaptive plan |
| Q3 rating filter | 1500+ Elo (verify metadata available at Phase 2B) | User: "go with your recommendations" |
| Q4 smoke gates | 1-iter + 5-iter both | User: "go with your recommendations"; 5-iter now also serves as mix-weight calibration probe |
| Q5 pod + timing | Prod, AFTER era4_v3 + two-snap MM eval + snap_0249 MM eval (Task #126) clear | User: "prod after mm evals clear" |

**Queue order**:
1. Wait era4_v3 (~3-5h remaining as of memo update)
2. Wait two-snap MM eval (~40 min as of memo update)
3. Run snap_0249 vs MMs (Task #126, ~30-40 min) — closes comparison triangle
4. **Then start AWR Phase 2A** (branch + skeleton) on prod

---

## Code references (for traceability)

| File | Lines | What |
|---|---|---|
| `ppo.py` | 594-709 | `_ppo_loss_batched_internal` — BC anchor pattern that AWR mirrors |
| `ppo.py` | 1446-1700+ | `ppo_update_batched` — entry point that needs AWR loop addition |
| `train_rl.py` | 1412-1425 | BC anchor loading pattern (reusable for π_behavior if Off-Policy PPO later) |
| `train_rl.py` | 1643-1657 | PPO update call site — new AWR flags plumb through here |
| `train_bc.py` | 81-200+ | BC training loop — reference for loss-from-memmap pattern |
| `dataset.py` | 22-145 | `MemmapDataset` — reusable directly via zero-pad path |
| `features.py` | 409, 653, 957, 1266, 1399 | Current obs dims — confirmed compat with memmap_v8 |
| `model_transformer.py` | 2456-2592 | `forward_ppo_sequence` — what AWR replay batch needs to match for shape |
| `data/datasets/memmap_v8/metadata.json` | — | BC v10 memmap dims (matches features.py) |
| `data/models/bc/v10_cloud_gen9/epoch_003.pt` | — | BC v10 checkpoint (already used as bc_ref) |
