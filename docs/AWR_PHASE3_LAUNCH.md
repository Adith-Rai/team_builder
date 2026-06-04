# AWR Replay Rehearsal Phase 3 — Launch & Monitoring Guide

**Status**: ACTIVE (launched 2026-06-04 16:05 UTC)
**Run**: `awr_phase3_v1` (PID 1971204 on prod), 30 iters, ~7.5 hr wall, ~$11

Operational doc for future sessions. For HYPOTHESIS + DESIGN see
`AWR_REPLAY_REHEARSAL_DESIGN.md`. For STRATEGIC CONTEXT (why we're doing
this at all) see `PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md`.

---

## Why we're doing this

S68 plateau diagnosis: BC v10 was trained on 1500+ Elo human replays so it
**SAW elite-play patterns** (synergistic team usage, sacrifice plays for
position, etc.). PPO from that init on **procedural teams** never gives those
elite patterns reward signal, so they decay over training. Result: model
becomes a competent generalist (~70-74% smart_avg) but **never refines elite
specialization** — explaining the empirically-observed ceiling and the gap
to true elite models (Minikazam, etc.).

AWR replay rehearsal = continue exposing the model to elite states/actions
DURING PPO. Each iter samples a small batch of (state, action, terminal_R)
from the SAME human-replay memmap BC v10 trained on, runs a binary AWR loss
(`-1[A > 0] × log π(a|s)` where `A = R - V_θ(s)`), and mixes that gradient
into PPO's optimizer. Effectively: "while you do PPO self-play, also keep
practicing elite human plays where you'd have agreed with the outcome."

This is the cheapest hypothesis test (replay rehearsal first, then template
teams, then metamon team_construction — per design memo ranking).

---

## What we expect (success criteria)

### Primary: MM-tier WR gain
Run external Elo ladder eval against:
- `snap_0139` (lr8e-5 record, 1178.4 Elo) = no-AWR baseline
- `awr_phase3_v1` snap_0009 / 0019 / 0029 = AWR-active model

Per design memo decision gate:
- **≥2pp MM-tier WR gain** vs snap_0139 (cheapest = LargeRL on metamon-competitive teams, n=500)
  → AWR validates direction → extend to 150-iter Phase 3-150 run
- **<1pp gain** → AWR signal too weak → see "If plateaus" below
- **regression (negative ΔWR)** → AWR harmful → see "If regresses" below

### Secondary signals during training
- `bc_kl` trajectory — smoke saw 0.167 → 0.216 over 7 iters, decelerating.
  Phase 3 expected plateau ≈ 0.22-0.28. **Yellow flag if >0.30**, **red flag if >0.40** (reward hack)
- `AWR loss` — should trend down as model converges toward replay distribution
- `adv_pos_frac` — should drop as V_θ catches up to replay terminal R
- `PPO KL` — must stay within `target_kl × 5 = 0.15` per-batch
- W/L ratio in self-play — drift will happen as policy changes but should stay roughly balanced (40-60%)

---

## Launch command (full)

Launch script lives at `/tmp/launch_awr_phase3.sh` on prod. The actual
invocation:

```bash
# On prod (port 47913, IP 195.26.233.30):
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)  # 151 entries, ~12 KB
INIT_CKPT=data/models/rl_v10/lr8e5_v1_flash/selfplay_v9_20260528_124244/snapshot_0139.pt
SERVERS="ws://127.0.0.1:9000/showdown/websocket,...9015..."  # all 16 BS

setsid nohup python -u train_rl.py \
  --init-from ${INIT_CKPT} \
  --out-dir data/models/rl_v10/awr_phase3_v1 \
  --n-iters 30 --warmup-iters 0 \
  --games-per-iter 1600 --turn-cap 300 --lr 8e-5 \
  --reward-style terminal \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --bc-anchor-ckpt data/models/bc/v10_padded_for_cis_dev.pt \
  --bc-anchor-coef 0.1 \
  --awr-replay-memmap data/datasets/human_v8_5k \
  --awr-mix-weight 0.05 --awr-batch-size 16 --awr-binary \
  --pool-anchors "${POOL_ANCHORS}" \
  --force-anchors ${INIT_CKPT} \
  --max-opponents-per-iter 10 \
  --external-adapters external_adapters_fishbowl_lr1e-4_v1.yaml \
  --n-ext-per-iter 5 \
  --cis --tier3 --tier3-minibatch-size 64 --bf16 \
  --mp-workers 90 --worker-cpu --pfsp-max-share 0.2 \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --servers ${SERVERS} \
  --snapshot-interval 10 --eval-interval 99 \
  --target-kl 0.03 --vf-coef 0.5 --max-grad-norm 0.5 --grad-accum 1 \
  --ent-coef 0.02 --adaptive-entropy \
  --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  </dev/null >/tmp/awr_phase3_v1.log 2>&1 &
```

### Key flag annotations

| Flag | Value | Why |
|---|---|---|
| `--init-from snapshot_0139.pt` | lr8e-5 record | Clean A/B comparator (1178.4 Elo). NOT from BC init — see [[awr-init-decision]] |
| `--reward-style terminal` | terminal | Per S68 reward-hack concern + lr8e-5 was terminal |
| `--awr-replay-memmap human_v8_5k` | 5k subset of R2 | Validated via Phase 2D smoke. Subset because prod disk = 46 GB free (R2 full = 104 GB) |
| `--awr-mix-weight 0.05` | 0.05 | Phase 2D calibration: AWR ≈ 2% of total loss, "stick at 0.05" bin |
| `--awr-batch-size 16` | 16 | Per design memo. ~0.5% of training transitions. See Task #133 for bump-to-64 consideration |
| `--awr-binary` | binary | `1[A > 0]` filter. Mirrors metamon's binary_rl.gin (SyntheticRLV2/Minikazam paradigm) |
| `--n-iters 30` | 30 | Design memo Phase 3 hypothesis-test scope. Extend to 150 if signal positive |
| `--warmup-iters 0` | 0 | snap_0139 already trained, no value-head warmup needed |
| `--eval-interval 99` | disabled | Pre-existing eval bug with comma-separated --servers. External Elo eval after instead |
| `--snapshot-interval 10` | 10 | snap_0009, snap_0019, snap_0029 for external eval |
| **NO `--packed`** | absent | AWR not compatible with packed in v1. ~3× slower per iter than fishbowl_prod but expected |
| **NO `--compile`** | absent | Per S62 refutation + AWR incompatibility |

---

## Monitoring

### Live monitoring (grep for meaningful events)
```bash
ssh prod "tail -F /tmp/awr_phase3_v1.log" | grep -E "^\[..:..:..\] Iter [0-9]+:|\[AWR \]|FATAL|Traceback|emergency"
```

### Iter cadence
~15 min/iter (no --packed). 30 iters = ~7.5 hr from launch at 16:05 UTC.

### Per-iter signal expected
```
[HH:MM:SS] Iter N: W/L/T=A/B/0 (XX.X%), N_steps, collect=XXXs, update=XXXs,
       pi=X.XXXX v=X.XXXX ent=X.XXXX kl=X.XXXX bc_kl=X.XXXX vs=cis(N=90,responded=90) pool=152
  [AWR ] loss=X.XXXX (scaled=X.XXXX, mix=0.05), adv_pos_frac=X.XXX, w_max=1.00, w_mean=X.XXX, grad_norm=X.XXX
```

### Health gates (kill if violated)
- `n_succeeded=0` → emergency
- `KL > 0.15` (target_kl × 5) → PPO early-stop triggers; one-off OK, multiple = bad
- `bc_kl > 0.40` → REWARD HACK zone; investigate
- `AWR step failed` (try/except WARN) → AWR bypassed for that iter; check next iter
- Process disappears with no `Training complete` → crash; check log for traceback

### Snapshot eval (after each snap_NNNN.pt saves)
Run on prod:
```bash
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
# Quick MM-tier check on cheapest MM (LargeRL):
python eval_mm_vs_smartbots.py \
  --bots none \
  --snapshots awr_phase3_iterN=data/models/rl_v10/awr_phase3_v1/selfplay_v9_*/snapshot_NNNN.pt \
  --n-games 500 --bot-concurrency 8 \
  --mm-startup-wait 90 \
  --team-set metamon-competitive \
  --out-json /tmp/awr_phase3_iterN_vs_mm.json
```

Compare to snap_0139's baseline (already captured in
`data/eval_artifacts/s68/snap_vs_mms_POST_INIT_iter139.json` =
49.6% vs LargeRL).

---

## Decision tree on results

### If positive (≥2pp MM-tier WR gain at snap_0029)
**Action: extend to 150 iters from snap_0029** (resume).
- More iters → more AWR exposure → stronger signal
- Run full Elo ladder eval comparing snap_0149 vs era4_chain
- If still gaining at iter 150: probably extend further OR start Off-Policy PPO arc
- Write up findings in `docs/AWR_PHASE3_RESULTS.md`

### If neutral (±1pp, within noise)
**Action: investigate why AWR signal is too quiet.**

Most likely cause: 16 episodes/iter is too small batch (only ~0.5% of
training transitions). Options:
- Bump `--awr-batch-size 64` (4×, matches PPO's `tier3-minibatch-size`)
- Re-extract larger replay subset (Task #132, sync full R2 to /dev/shm
  + extract 10k+ episodes)
- Bump `--awr-mix-weight 0.10` (calibration table allows this if AWR loss
  fraction was <2% in Phase 3 — check actual fraction)

Re-run 30 iters with bumped params. If still neutral after 2-3 variants:
hypothesis weakening, see "If regresses" plan.

### If regresses (negative MM-tier ΔWR or training destabilization)
**Action: hypothesis weakening, escalate or reconsider.**

Diagnostic order:
1. Check `bc_kl` trajectory across run — did it exceed 0.30 (drift) or
   0.40 (reward hack)? If yes, AWR pulled policy too hard toward BC v10
   distribution → BC anchor coef 0.1 was insufficient. Try `--bc-anchor-coef
   0.15` or `0.20` to counter-anchor.
2. Check WR trajectory — drop in self-play indicates AWR push made model
   worse against pool. Could mean replay distribution doesn't match
   self-play distribution well.
3. Switch to exp(A/β) variant: drop `--awr-binary`, add `--awr-beta 1.0`.
   May find more nuanced signal than binary filter.
4. Escalate to Off-Policy PPO (see `REPLAY_REHEARSAL_AWR_VS_OFFPOLICY_PPO.md`).
   Proper RL update with IS ratio, can both reinforce AND suppress actions
   (binary AWR only reinforces). More implementation work (need behavior
   policy estimate) but more powerful.
5. If Off-Policy PPO also fails: hypothesis "replay rehearsal reinforces
   fading elite patterns" is REFUTED. Move to template teams (Task #124
   metamon team_construction) or sourced elite teams (Task #123).

### If training destabilizes (NaN losses, smart-avg crash, KL explosion)
**Action: stop run, diagnose, restart with tighter constraints.**
- Drop `--awr-mix-weight` to 0.02-0.03
- Or bump `--bc-anchor-coef` to 0.2 (stronger anchor against drift)
- Or both
- Then re-fire shorter (10-iter) smoke to verify stability before retrying

---

## Implementation notes for future sessions

### Pool size
The 151-entry pool is from `fishbowl_prod_lr1e-4_v1/config.json["pool_anchors"]`.
~12 KB string. Stored at `/tmp/phase3_pool_anchors.txt` on prod for reuse.
Pool grows by 1 per snapshot (interval 10 → +3 by iter 30, so final pool
= 154 entries).

### Per-iter speed
~15 min/iter without --packed (collect ~4 min, update ~10 min, AWR step
~0.5 min). With --packed: would be ~5-7 min/iter but AWR not compatible.

### External eval setup
The in-loop eval bug (`--servers` comma-list parsing) is documented as
"pre-existing poke-env URL-parse issue with multi-server eval." Future
fix would let us re-enable in-loop eval. For now, external Elo runs are
the source of truth. See `eval_mm_vs_smartbots.py` + `eval_elo_ladder_cis_v2.py`.

### Pod state hygiene before launch
1. `pkill -9 -f train_rl.py.*<old>` (kill prior train procs)
2. `pkill -9 -f battle_server.js && sleep 2 && bash /tmp/launch_bs.sh && sleep 6` (restart BS)
3. Verify `pgrep -fc battle_server.js == 16` before launching
4. THEN launch

Skipping the BS restart leads to asymmetric worker stalls (per
`feedback_battle_server_restart_after_kill.md` memory note).

### Smoke validation (Phase 2D summary)
7 iters from snap_0139 on prod, validated:
- AWR via `forward_ppo_sequence` (SAME forward path as PPO update;
  verified equivalence test `test_forward_paths_equivalence.py` shows
  the two forward paths DIVERGE by 1.89e-3 — that's why AWR uses
  forward_ppo_sequence to match PPO's update distribution)
- mix=0.05 → AWR contributes mean 2.0% of total loss across iters
- WR balanced 50.7%, KL within target, bc_kl creep decelerating
- 5 critical bugs caught + fixed (subset format, forward path, episode_index, eval comma-list, BS restart hygiene)

---

## Files

- This doc: `docs/AWR_PHASE3_LAUNCH.md`
- Design: `docs/AWR_REPLAY_REHEARSAL_DESIGN.md`
- Theory: `docs/REPLAY_REHEARSAL_AWR_VS_OFFPOLICY_PPO.md`
- Strategic context: `docs/PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md`
- Memory pointer: `memory/project_awr_replay_rehearsal_design.md`
- Smoke artifacts: `data/eval_artifacts/s68/awr_smoke_v2e*`
- Code: `pokemon-ai-starter/pokemon-ai/src/awr_replay.py`
- Subset script: `pokemon-ai-starter/pokemon-ai/src/subset_memmap.py`
- Equivalence test: `pokemon-ai-starter/pokemon-ai/src/test_forward_paths_equivalence.py`
- Branch: `feat/replay-rehearsal-awr`
