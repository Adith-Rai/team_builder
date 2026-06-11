#!/bin/bash
# RUN #9 — heuristic-opp diversity ("guardrails" framing)
# Branch: master (verify HEAD has heuristic adapter + v2 bots)
#
# Reframe (S68 2026-06-10): we discovered Run #7's no-anchor collapse wasn't
# just exploration valley — the SP-pool members all share BC-derived
# decision style (BC anchor was used in 5 of 6 contributing runs), so when
# Run #7 developed setup-spam, the pool couldn't punish it well. AWR alone
# isn't doing meaningful BC-pull work (measured ~24 Elo of micro-improvements
# only, no decision-pattern impact in Run #5 vs Run #6 replay analysis).
#
# Run #9 hypothesis: heuristic bots provide CATEGORICALLY-DIFFERENT decision
# processes (rule-based, not BC-derived). 5 v2 bots designed S68 2026-06-10
# all at eval-bot-tier Elo (949-1043) with distinct specialties:
#   GreedySEv2          (1043 Elo) — greedy super-effective attacker
#   SetupThenSweepv2    (1031 Elo) — forced setup at HP>=70% + sweep
#   SwitchAwareEscapev3 (1012 Elo) — offensive-pivot specialist
#   AntiSetupBot         (968 Elo) — anti-setup punisher
#   HazardSensev2        (949 Elo) — hazards-first utility
#
# Design (per user "guardrails" framing 2026-06-10, REVISED 2x):
#   - ADDITIVE — don't reduce existing signal vs Run #7 (5 self stays at 5)
#   - SEPARATE heuristic sub-pool with own PFSP (S68 code change). Prevents
#     MMs (~15% model WR) from dominating single-pool PFSP and starving
#     heuristics (~30-50% WR, closer to PFSP target).
#   - +4 heuristic slots (3 PFSP-weighted + 1 random) via --n-heur-per-iter 4
#   - --n-ext-per-iter 4 (was 5: MCTS deferred — see docs/TODO_MCTS_RUN9.md)
#   - --n-heur-per-iter 4: NEW dedicated heuristic slots
#   - --max-opponents-per-iter 14: 1 force + 5 self + 4 ext + 4 heur
#   - --games-per-iter 2240: 160 games per opp (matching Run #7 per-opp density)
# Yaml has 4 MMs + 13 non-eval heuristic bots. PFSP within heur pool
# naturally downsamples weak bots; random slot ensures variety.
#
# ⚠️ MCTS DEFERRED 2026-06-10. See docs/TODO_MCTS_RUN9.md for the diagnosis
# (per-worker MCTS executor serialization + new InvalidWeight panic) +
# concrete investigation hooks before re-adding to Run #10+.
#
# Cost projection vs Run #7:
#   Per-iter: 40% more games, but heuristics are ~3s/game vs MM ~5s/game
#   Net iter cost: ~30% increase. ~13-14 min/iter vs Run #7's ~10 min.
#   Total wall: ~45 hr for 200 iters (~$70 vs Run #7's $50).
#
# Differences from Run #7 (no anchor + AWR + 30% syn):
#   - --external-adapters: phase3_full_v2_heur.yaml (was full_v1)
#   - --n-ext-per-iter 9 (was 5)
#   - --max-opponents-per-iter 14 (was 10)
#   - --games-per-iter 2240 (was 1600)
#
# Everything else IDENTICAL to Run #7:
#   BC v10 init, NO anchor, AWR binary mix 0.15, syn 30%, lr 8e-5,
#   compute stack (--cis --tier3 --bf16 --tier3-minibatch-size 64),
#   90 mp-workers, cis-min-batch 32, cis-timeout-ms 50.

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_run9_heur_diversity_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5/#6/#7)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check bundles exist
if [ ! -f "${SYN_HL}.teampack" ] || [ ! -f "${SYN_GL}.teampack" ]; then
  echo "ERROR: syn team bundles missing. Run R2 download from CLOUD_RUNBOOK.md."
  exit 1
fi

# Sanity check new yaml exists
if [ ! -f "external_adapters_phase3_full_v2_heur.yaml" ]; then
  echo "ERROR: heuristic-yaml missing. Should be in src/ after git pull."
  exit 1
fi

setsid nohup python -u train_rl.py \
  --init-from ${INIT_CKPT} \
  --out-dir ${OUT_DIR} \
  --n-iters 200 --warmup-iters 5 \
  --games-per-iter 2240 --turn-cap 300 --lr 8e-5 \
  --lam 0.95 \
  --reward-style terminal \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --syn-team-dirs ${SYN_HL}:0.6,${SYN_GL}:0.4 \
  --syn-team-pct 0.30 \
  --syn-intra-asymmetric-rate 0.30 \
  --top-asymmetric-rate 0.20 \
  --awr-replay-memmap data/datasets/human_v8_5k \
  --awr-mix-weight 0.15 --awr-batch-size 16 --awr-binary \
  --pool-anchors "${POOL_ANCHORS}" --force-anchors ${INIT_CKPT} \
  --max-opponents-per-iter 14 \
  --external-adapters external_adapters_phase3_full_v2_heur.yaml \
  --n-ext-per-iter 4 --n-heur-per-iter 4 \
  --cis --tier3 --tier3-minibatch-size 64 --bf16 \
  --mp-workers 90 \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --servers ${SERVERS} \
  --snapshot-interval 10 \
  --target-kl 0.03 --vf-coef 0.5 --max-grad-norm 0.5 --grad-accum 1 \
  --ent-coef 0.02 --adaptive-entropy \
  --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  </dev/null >/tmp/run9_heur_diversity_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #9 HEUR DIVERSITY LAUNCHED ==="
echo "PID: ${PID}"
echo "Log: /tmp/run9_heur_diversity_v1.log"
echo "Out: ${OUT_DIR}"
echo ""
echo "Diff from Run #7:"
echo "  --n-ext-per-iter 5   (UNCHANGED — MMs/MCTS slot allocation preserved)"
echo "  --n-heur-per-iter 4   (NEW — dedicated heuristic pool with own PFSP)"
echo "  --max-opponents-per-iter 15   (was 10: 1 force + 5 self + 5 ext + 4 heur)"
echo "  --games-per-iter 2240   (was 1600: keeps per-slot games at 160)"
echo "  --external-adapters full_v2_heur.yaml   (was full_v1, adds 13 heur bots)"
echo ""
echo "13 heuristic bots (all non-eval):"
echo "  v2 strong (5, ~950-1043 Elo): GreedySEv2, SetupThenSweepv2,"
echo "      SwitchAwareEscapev3, AntiSetupBot, HazardSensev2"
echo "  v2 weaker (2, ~720-880 Elo): StrategicV2, SwitchAwareEscapeV2"
echo "  Raw originals (4, ~730-830 Elo): GreedySE, HazardSense,"
echo "      SwitchAwareEscape, SetupThenSweep"
echo "  poke-env baselines (2, ~400-720 Elo): RandomPlayer, MaxBasePower"
echo "PFSP within heur pool downsamples weak bots; 1 random slot adds variety."
echo ""
echo "200 iters @ projected ~13-14 min/iter = ~45 hr wall"
echo ""
echo "Watch for at iter 0:"
echo "  - '[cis-w*] paired-pool teambuilder active' (syn config)"
echo "  - 'Spawning mm-*' x 12 (4 MMs x 3 instances)"
echo "  - PFSP-ITER includes heur-* opponents in active set"
echo "  - smart_avg trajectory at iter 9, 19, 29..."
echo ""
echo "Hypothesis test signals:"
echo "  - If smart_avg STAYS ABOVE 50 (vs Run #7 dropping to 22): heuristics work"
echo "  - If MM trajectory holds positive: not over-rotating to heuristic-style"
echo "  - If snapshot_0149/_0209 trajectory positive: BC-distance manageable"
