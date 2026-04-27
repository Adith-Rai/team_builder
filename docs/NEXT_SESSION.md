# NEXT_SESSION.md — Project Handover

**Last updated: 2026-04-26 (Session 42 — external-opponent integration VALIDATED end-to-end; protocol bridge bugs all fixed)**

This is the canonical reference for resuming work on this project. It's self-contained —
read this top-to-bottom and you should have full context to execute every pending task.

Supporting documents:
- `docs/EXTERNAL_OPPONENTS_PHASE2.md` — **READ THIS** for the protocol-bug postmortem and reproducer
- `docs/METAMON_LEARNINGS.md` — Session 37 Metamon architecture study + recommendations
- `docs/RESEARCH.md` — architecture research, published system comparisons, experiment order
- `docs/STATUS.md` — full historical narrative if deep context needed (long, usually skippable)
- `docs/CLOUD_DEPLOY.md` — cloud migration plan

---

## Session 42 status (READ THIS FIRST)

**External-opponent integration is now WORKING.** All four opponent paths
play a complete battle to completion against a poke-env 0.10 sender on
the local battle_server, validated with `diag_cross_venv.py`:

| Path | Adapter | First battle clean? |
|---|---|---|
| Self-play | `SelfPlayOpponent` (in-process) | ✓ |
| Foul Play MCTS core | `mcts` / `pokeengine` (in-process via poke-engine) | ✓ |
| Real Foul Play | `foulplay` subprocess in `foul_play_venv` | ✓ (19s, 100ms MCTS) |
| Metamon (Minikazam, 4.7M) | `metamon` subprocess in `metamon_venv` | ✓ (1.7s) |

The Session 39 conclusion ("subprocess design hit a wall, need to rewrite
as in-process Players") was **wrong**. The wall was server-side protocol
bugs in `battle_server.js`, not architectural. Once fixed (Session 42),
the original subprocess design completes battles cleanly. The skeleton
we built (process manager, QueueTeambuilder, MultiSourceTeambuilder,
PoolEntry) is all in production use.

**Nine protocol bugs fixed this session — see `docs/EXTERNAL_OPPONENTS_PHASE2.md`
for the postmortem and a 5-min reproducer recipe.** Short version:
1. Per-recipient framing in `pumpPlayer` (5-frame Showdown-faithful for
   FP/MM, bundled for poke-env 0.10).
2. `/choose`, `/team`, `/switch`, `/move` all strip trailing `|<rqid>`
   before forwarding to BattleStream (real-Showdown clients always append
   it; BattleStream rejects).
3. `isShowdownFaithful` checks display name (with dash) since toId form
   strips the `MM-` dash.
4. `_factory_metamon` sets `TORCHDYNAMO_DISABLE=1` on Windows (Triton has
   no Windows wheels; Metamon's `torch.compile` crashes otherwise).
5. (Session 39, already committed) `/challenge` PM uses Showdown-standard
   8-pipe / 9-split-field format.
6. **Multi-battle: `/leave <battle-tag>` echoes `>tag\n|deinit`** so FP's
   `leave_battle` returns. Without this FP hangs after every battle. FP
   sends /leave as a global command (`|/leave battle-tag`, empty room
   prefix) — fix is in the global /leave branch, not per-battle one.
7. **Multi-battle: `cleanupBattle` re-emits pending `|pm|/challenge`** to
   users who just became idle. Pre-fix, /pms sent during a battle were
   silently consumed by `pokemon_battle` and never reached `accept_challenge`.
8. **Multi-battle: drop `|updatechallenges|`, send only `|pm|/challenge`**.
   poke-env (in metamon's 0.8.3.3 fork) registers the challenger on
   `_challenge_queue` from BOTH frames, double-populating the queue.
   Metamon's iter 1 consumed one and /accepted; iter 2 consumed the
   duplicate and sent a stale /accept that battle_server rejected with
   "No pending challenge". FP only reads `|pm|`, so it's fine without
   `|updatechallenges|`.
9. **Multi-battle: bump Metamon's openai_api 50s idle timeout to 1 hour**
   via class attribute override (`_INIT_RETRIES = 7200`,
   `_TIME_BETWEEN_RETRIES = 0.5`) on `AcceptChallengesOnLocal`. Default
   raised `RuntimeError("Agent is not challenging")` 50s after each battle
   ended if the next /challenge didn't arrive in time — PFSP gaps blew
   that. Override picked up via MRO when openai_api does
   `self._INIT_RETRIES`.

**Full PPO smoke validated.** `train_rl.py --external-adapters
external_adapters_all3_smoke.yaml --games-per-iter 9 --max-concurrent 3
--n-iters 1 --init-from <sp0229.pt>` completes cleanly:
`Iter 0: W/L/T=3/6/0 (33.3%), collect=65s, update=2s, vs sp0229=2/3
mcts-fast=1/2 foulplay-100ms=0/2 metamon-minikazam=0/2`. All 4 routes
(self-play, in-process mcts, FP subprocess, MM subprocess) drove the
real PFSP collection loop end-to-end.

**Bonus:** `battle_server.js` now prefixes log lines with
`HH:MM:SS.mmm` timestamps for future protocol debugging.

### Session 42 — Throughput pass (Task #19, partially complete)

**Full opponent pool YAML** at `external_adapters_full_pool.yaml`:
- `mcts-fast` (in-process MCTS via poke-engine, weight=1.0)
- `foulplay-100ms-1..4` (4× FP subprocesses, each weight=0.25, sums to 1.0)
- `mm-minikazam` / `mm-smallrl` / `mm-smallil` / `mm-mediumrl` (4 different
  Metamon variants, each weight=0.25). Diverse policies: small RL, medium
  RL, BC-only, gen9-tuned RL.

**VRAM measured (Session 42, replacing earlier 870MB-per-MM estimate that
was actually working-set RAM, not GPU):**
- Minikazam (4.76M params): **173 MB GPU**
- SmallRL (13.9M): **218 MB**
- SmallIL (13.7M): **215 MB**
- MediumRL (50.5M): **530 MB**
- Total 4 MMs: **~1.14 GB** (comfortable on 6GB GPU alongside 3-4GB trainer)

**Wave parallelism finding (THE key knob): `--servers 9000,9000,9000,9000`**
creates 4 server-pool entries all pointing to the same battle_server.
`rl_collection.py` processes opponents in waves of `n_servers`, so this
yields 4× wave parallelism on a single battle_server with **zero code
changes**. Validated 1-iter smoke: 18 games in **124s** (vs 236s baseline
with single-slot wave) → **1.9× speedup**. PROF wave 0 hit batch=1.4
peak=5 → InferenceBatcher actually batching now.

**6-slot and 10-slot still fail (after one fix).** Diagnosis:

1. **Login-time |pm|/challenge race (FIXED, bug #10).** When the trainer
   issues N concurrent /challenges to N opponents, FP/MM subprocesses
   may not have finished their `/trn` handshake yet. battle_server
   silently dropped the |pm| (no ws to send to). Fix: in the /trn
   handler, after registering the user, iterate `pendingChallenges` for
   any entries targeting this user and emit |pm| directly. Validated at
   4-slot (the existing ceiling), still safe — fires only when there's
   a pending challenge.

2. **Still broken at 6+ slots (NOT FIXED).** Two separate issues:
   - **FP cascading restart starvation:** at 10-slot, FP1 hit
     `websockets.ConnectionClosedError: no close frame received or sent`
     mid-handshake, ExternalOpponentManager auto-restarted it, but its
     enqueued team file had been consumed by the first instance's
     `yield_team()`. The restarted FP1 sat on iter 1 with empty queue.
   - **MM `_challenge_queue` AttributeError:**
     `AttributeError: 'AcceptChallengesOnLocal' object has no attribute
     '_challenge_queue'` from poke_env's `_handle_challenge_request`.
     The login-time |pm| arrives during MM setup before the agent's
     `_challenge_queue` is bound, and the handler crashes. This is in
     metamon's poke-env fork (0.8.3.3) — not easily fixable from our side.

**4-slot remains the validated production ceiling.** 1.9× speedup over
single-slot, all 9 external entries play battles cleanly. Smoke result:
`Iter 0: W/L/T=2/16/0 (11.1%), 18 games, collect=141s` (with new login-
resend code in place — slightly slower than the 124s pre-fix run, within
PFSP noise).

### 5-iter PPO smoke (Session 42, full pool stability validation)

To confirm the full pool isn't just one-shot stable, ran 5 consecutive
iters with `external_adapters_full_pool.yaml` + `--servers 9000,9000,9000,9000`
+ `--games-per-iter 18 --max-concurrent 6 --eval-interval 999` (skip evals).
All 5 iters reached PPO update; total wall-clock ~13 min:

| Iter | W/L (%)      | Collect | Update | Notes                          |
|------|--------------|---------|--------|--------------------------------|
| 0    | 8/10 (44%)   | 123s    | 6s     | clean                          |
| 1    | 4/14 (22%)   | 123s    | 7s     | clean                          |
| 2    | 7/11 (39%)   | 162s    | 8s     | clean                          |
| 3    | 8/10 (44%)   | 125s    | 6s     | clean                          |
| 4    | 4/13 (24%)   | 235s    | 6s     | poke-engine PanicException     |

**Iter 4's outlier**: one `mcts-fast` battle hit
`PanicException('Encore should not be active when last used move is not
a move')` — a niche edge case in the poke-engine Rust library validating
illegal sim states. Trainer's `wait_for` caught it, logged
`[WARN] Timed out vs mcts-fast after 2 games`, moved on. Lost 1 game
out of ~90 (1% failure rate). Recovery is automatic — no manual
intervention needed.

**Stability evidence:**
- Subprocess-side: no FP/MM crashes or auto-restarts across all 5 iters
- Coordinator-side: 2 login-time resends (early iter), 20 cleanup-time
  resends (~4/iter, normal for the multi-battle flow)
- Memory: stable, no leaks across iters
- PFSP win-rate evolution: visible across iters (e.g. mm-smallrl was
  hard early, sampled more later by `(1-wr)^2` weighting)

**Conclusion: full pool config is production-ready for long PPO runs.**
Per-iter cost extrapolation: 200 games/iter ≈ 200/18 × 130s avg = ~24
min/iter. A 50-iter run = ~20 hr. A 100-iter run = ~40 hr. Plan
accordingly.

**Caveat: SmallRLGen9Beta and other FlashAttention-required variants
crash on Windows** (Triton-less). The 4 variants in the YAML use
VanillaAttention fallback successfully. Adding more variants requires
checking they don't hard-require flash attention.

### Production runbook (validated end of Session 42)

Full PPO training run with the multi-opponent pool. Two terminals,
copy-pasteable.

**Pre-flight (run once before each fresh training run):**

```bash
# 1. Kill any stale processes from previous runs
powershell -Command "Get-Process node, python -ErrorAction SilentlyContinue | Stop-Process -Force"

# 2. Clean external opponent team queues (otherwise stale teams from
#    aborted runs accumulate)
rm -rf C:/Users/raiad/OneDrive/Desktop/team_builder/data/external_team_queue/foulplay-100ms-*/
rm -rf C:/Users/raiad/OneDrive/Desktop/team_builder/data/external_team_queue/mm-*/

# 3. Verify the init checkpoint exists (this is the Session 39 PPO record;
#    swap path if resuming from a different snapshot)
ls C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt

# 4. Verify the full-pool YAML exists
ls C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/external_adapters_full_pool.yaml
```

**Terminal 1 — battle_server (single instance handles 4-slot wave):**

```bash
C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe \
  C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js \
  --port 9000 \
  2>&1 | tee C:/Users/raiad/OneDrive/Desktop/team_builder/logs/external/battle_server.log
```

Wait for `[battle_server HH:MM:SS.mmm] Listening on port 9000` then leave it running.

**Terminal 2 — production training run:**

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
  --device cuda --servers 9000,9000,9000,9000 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 6 --n-iters 100 --warmup-iters 0 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --adaptive-entropy --early-stop --win-rate-mode ema \
  --eval-interval 20 \
  --out-dir data/models/rl_v9_full_pool \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  --external-adapters external_adapters_full_pool.yaml \
  2>&1 | tee training.log
```

The trainer auto-spawns FP and MM subprocesses via `ExternalOpponentManager`
based on the YAML, so no separate launch step. Spawn takes ~30s
(Metamon model loads dominate; trainer waits via `wait_until_ready`).

**Why each flag:**

| Flag | Purpose |
|------|---------|
| `--init-from <pt>` | Fresh PPO state from this checkpoint (separate optimizer state). Use `--resume <pt>` instead to continue an interrupted run with optimizer state preserved. |
| `--servers 9000,9000,9000,9000` | THE throughput knob. 4 server-pool slots all pointing at the same battle_server → 4× wave parallelism. Tested up to 4. **6+ stalls** (see deferred bugs above). |
| `--fp16` | Mixed precision on inference + PPO. ~2× speedup, no quality regression measured. |
| `--pipeline` | Background collector overlaps next iter's collection with current iter's PPO update. Saves the update wall-time on every iter (60–80s at 200 games). Costs ~1GB extra RAM (model copy on CPU). **Recommended on for production runs.** |
| `--games-per-iter 200` | Standard for our PPO scale. Smaller = faster iters but noisier gradients. |
| `--max-concurrent 6` | Per-opponent V9RLPlayer concurrent battles. With 4-slot wave × 6 = up to 24 concurrent battles. Higher works on bigger GPUs but doesn't help here. |
| `--n-iters 100` | 100-iter run ≈ 40 hr at the measured ~24 min/iter. Adjust to budget. |
| `--lam 0.95 --ent-coef 0.02` | Validated hyperparams from Session 39 (the smart_avg-64% record). |
| `--adaptive-entropy --early-stop` | Safeguards from Session 35. Prevent entropy collapse. **Always on** for long runs. |
| `--win-rate-mode ema` | EMA over last 50 games per opponent for PFSP. Better than cumulative for non-stationary policies. |
| `--eval-interval 20` | Eval against the 4 fixed eval bots every 20 iters. Set to 999 for smokes (skip evals entirely). |
| `--external-adapters <yaml>` | Wires the 9 external opponents into the snapshot pool with PFSP weights from the YAML. |

**What to watch in `training.log`:**

- **Healthy iter line** (one per iter, ~24 min apart):
  `[HH:MM:SS] Iter N: W/L/T=W/L/0 (X%), N steps, collect=Ts, update=Ts, pi=... v=... ent=... kl=... vs=<per-opp> pool=10`
- **Resends (normal, ~4/iter at 4-slot):**
  `[battle_server HH:MM:SS.mmm] Resent pending challenge X -> Y after battle cleanup`
- **Anomalies — investigate:**
  `[WARN] Timed out vs <opp>` (1–2/iter is OK from poke-engine panics; >5/iter = real problem)
  `Traceback` / `FATAL` (anywhere)
  `KL early stop: epoch 0` on every iter (means batch is too small or learning rate too high)

**Resume an interrupted run:**

```bash
# Find latest snapshot in the run dir
ls -t data/models/rl_v9_full_pool/selfplay_v9_*/snapshot_*.pt | head -1
# Then change --init-from to --resume in the Terminal 2 command
```

`--resume` preserves PPO optimizer state (Adam momentum/variance), the
PFSP pool with its win-rate stats, the run dir. `--init-from` gives a
fresh start.

**Stop cleanly:**

`Ctrl-C` the trainer in Terminal 2. The `ExternalOpponentManager` will
SIGTERM all FP/MM subprocesses on shutdown. Final checkpoint saves to
`<out_dir>/.../final.pt`. Battle_server in Terminal 1 keeps running
(no state to clean up); Ctrl-C separately or leave for next run.

**Throughput levers in priority order if 24 min/iter is too slow:**

1. `--pipeline` (already in command above; saves ~60s/iter, free).
2. Lower FP `search_time_ms` from 100 to 50 in `external_adapters_full_pool.yaml` (FP stays strong, ~2× faster per battle).
3. Replace 2 of 4 `foulplay-100ms-N` entries with additional `mcts-fast` entries — in-process, no subprocess serialization.
4. `--mp` flag for multiprocess collection (untested with current setup, exists in code).
5. **DO NOT** push `--servers` past 4 slots (cascading FP restart + MM `_challenge_queue` race; see "deferred bugs" section above).

### TL;DR for next session

**Two pending items**, in priority order:

1. **Real long PPO run with the validated full-pool YAML.** Use the
   command in section above with `--servers 9000,9000,9000,9000
   --external-adapters external_adapters_full_pool.yaml`. Extrapolating
   from 18 games / 124s, a 200-game iter ≈ 23 min. Tight but workable.
   Optionally: `--pipeline` to overlap PPO update with next collection
   (saves ~5s/iter). Tune `--games-per-iter` if collection time is
   prohibitive — 100 games/iter gives ~12 min iters.

2. **Optional**: investigate the 6+ slot wave stall. Likely either
   (a) `_play_one_opponent`'s send_challenges call concurrency causing
   /pm crossings on battle_server; (b) battle_server's pendingChallenges
   map being keyed by (challenger,) so simultaneous challenges from N
   different RL players to the same opponent overwrite each other (no
   — different challengers, different keys); (c) ws backpressure on the
   single battle_server from N concurrent senders. Diagnostic: enable
   `BS_TRACE_USER=foulplaybot1` and watch where the deadlock forms.

3. **Incremental Elo on the 6-snapshot "Useful" set** (~2.5 hr,
   orthogonal): New BC + sp_0029 + sp_0099 + sp_0159 + sp_0219 + sp_0229.
   Adds incrementally to `data/eval/elo_exp5_FINAL.json`. Anchors on
   sp2979 (Elo 1058, still on disk). Settles whether the Session 39
   smart_avg gain (64% all-time peak) is a real Elo gain.

### PPO run on the human-data BC — DONE

Run dir: `data/models/rl_v9/selfplay_v9_20260425_062416/` (resumed from
the prev run's snapshot_0029). 200 iters total (iters 30 through 229).
Settings: `lr=3e-5`, `lam=0.95`, `ent-coef=0.02`, `--adaptive-entropy`,
`--win-rate-mode ema`, procedural teams. **Note: lr=1e-4 caused 80%+
KL discard in the original attempt; lr=3e-5 is the right starting point
from a 45% BC.** All eleven 20-iter evals plus the manual final eval:

| Iter | SH | SmDmg | Tact | Strat | smart_avg |
|---|---|---|---|---|---|
| 19 | 52 | 56 | 48 | 55 | 53.0 |
| 39 | 70 | 58 | 62 | 58 | 62.0 |
| 59 | 57 | 54 | 60 | 56 | 56.0 |
| 79 | 66 | 66 | 56 | 56 | 61.0 |
| 99 | 63 | 56 | 57 | 60 | 59.0 |
| 119 | 61 | 63 | 48 | 60 | 58.0 |
| 139 | 59 | 61 | 60 | 60 | 60.0 |
| 159 | 60 | 68 | 60 | 65 | 63.0 |
| 179 | 62 | 60 | 59 | 64 | 61.0 |
| 199 | 62 | 57 | 57 | 54 | 58.0 |
| **219** | **61** | **68** | **60** | **66** | **64.0** ← all-time peak |
| 229 (final, manual) | 67 | 61 | 56 | 69 | 63.4 |

Best snapshots: `sp_0219.pt` (peak 64%) and `sp_0229.pt` (final 63.4%).
Both are statistically tied; either could be best in actual Elo terms.

**Internal PFSP win rates** (after fixing my misread of the EMA file
format — `[wr*eff_games, eff_games]`, divide by 50 for %):
- vs **sp2979** (the historical Elo 1058 ceiling): we win **63%**
- vs **BC_base** (our new 45% BC): 86%
- vs hardest recent self-play (sp_0189): 35%
- vs prev-run snapshots (sp_0004…sp_0019): 75-76%
**i.e. we are solidly past sp2979 in actual head-to-head play, even
though smart_avg vs the bot anchors plateaued at ~60%.** Smart_avg has
likely saturated against the eval bots; the policy is genuinely
stronger than the metric resolves.

### External opponents — SKELETON DONE, ADAPTER WORK PENDING

What's committed (~5 commits, see git log):
- `external_opponent_manager.py` — process manager + YAML config (still
  useful if/when a bot doesn't have a clean Python entrypoint)
- `external_opponents_example.yaml` — config schema
- `metamon_local.yaml` — agents config for `metamon.rl.self_play.serve_model`
- `team_generator.py` — added `MultiSourceTeambuilder` + `StaticTeamPool`
- `battle_server.js` — `e01a37f` patches the /challenge PM to standard
  Showdown 9-field format (was 5-field inline; broke Foul Play's parser)

What we *learned* and *did not* commit upstream:
- `foul_play_ref/fp/websocket_client.py` needs a 1-line patch to do
  case-insensitive `_to_id`-style username comparison. Documented in
  `docs/EXTERNAL_OPPONENTS_PHASE2.md` step 1. Lives only in the local
  clone of foul_play_ref/, since that's an upstream repo we don't
  vendor.
- `poke-engine` (FoulPlay's MCTS dependency) builds via PEP 517 + Rust
  toolchain transparently; no manual Rust install needed.

What's broken / re-thought:
- Subprocess-bot + send_challenges design doesn't drive battles to
  completion. Phase 2 runbook now leads with the recommended adapter
  approach (Python Player wrapping `poke-engine` directly).

### Analysis tooling — DONE

`analyze_eval.py` extended with 5 features (3 commits this session):
- `--iter-trajectory <run_dir>` — per-iter playstyle table across
  `replays_iter*/` subdirs
- `--by-opponent` — split trajectory rows by opponent bot
- `--team-usage` — lead mon, send-out frequency, faint order
- `--decision-quality` — attacks-into-immune, switches-into-SE,
  setup-at-low-HP, recovery-at-full-HP rates
- `--human-baseline N` — stream N HF replays at >=1500 Elo, compute
  same playstyle profile, add as comparison column

Findings from running these on the current PPO run (iter 39 → iter 179):
- Voluntary switches halved (12% → 6.5%) — model committing more
- SE rate +4.5pts (49 → 53.5%) — better type targeting
- Setup-at-low-HP −5pts (19 → 14%) — fewer wasted setups
- **Surprise:** immune-attack rate went UP (11.4% → 13.5%). Worth
  flagging — possibly because the policy converged on a smaller set of
  preferred moves and some of those happen to hit immunities. Doesn't
  block training but is a real failure mode to investigate.

### Pipeline + infrastructure fixes (committed)

- `replay_to_memmap.py`: stale `make_obs_mask_and_slots` import removed;
  `MemmapV8Writer.finalize()` now actually trims raw .npy files instead
  of just claiming to. Reclaimed 63.67 GB from the existing
  `human_v8_100k/` memmap (174→111 GB).
- BC reshape on `human_v8_100k` produced `best.pt` at smart_avg 45.1%
  (epoch 2, 14.28M params, 2:1 temporal:spatial reshape). This is the
  BC base for the PPO run above.

---

## Session 38 status (kept for context)

**Session 38 completed two workstreams: (a) BC reshape trained to completion on the old
bot-generated memmap, reaching smart_avg 21.2% at epoch 4 (below the historical 22-26%
ceiling — a small regression); (b) regenerated the training data from 100k ≥1500-Elo
human Showdown replays, with current 109/30 features (type-effectiveness dims now present).
The reshape hypothesis is untested against the better data source, and that's what next
session should do.** Full details below.

### TL;DR for next session
1. Launch BC reshape training on the new human memmap (`data/datasets/human_v8_100k/`)
   — same reshape config, same 10 epochs. See "Next concrete step" below.
2. If `best.pt` beats 22-26% smart_avg, the capacity reshape + human data combination
   works. If flat at ~21%, we're at the BC ceiling and PPO is the next lever.

### BC reshape run on bot data — DONE (final result)

Run dir: `data/models/bc/v8_bc_20260423_124909/` (restarted after the mid-run BSOD kill).
All 10 epochs completed. Best at **epoch 4: smart_avg 21.2%** (SH=21, SmartDmg=20,
Tactical=22, Strategic=22). Per-epoch curve showed classic BC overfit signature:
val_acc climbed monotonically 0.671→0.713 while smart_avg plateaued at ~20%.

| Epoch | SH | SmDmg | Tact | Strat | avg | val_acc |
|---|---|---|---|---|---|---|
| 0 | 21 | 22 | 18 | 18 | 20.0 | 0.671 |
| 1 | 18 | 24 | 20 | 18 | 19.6 | 0.675 |
| 2 | 27 | 23 | 16 | 16 | 20.6 | 0.683 |
| 3 | 20 | 19 | 20 | 20 | 19.8 | 0.699 |
| **4** | **21** | **20** | **22** | **22** | **21.2** | **0.706** ← best.pt |
| 5 | 16 | 26 | 22 | 19 | 20.6 | 0.706 |
| 6 | 15 | 17 | 22 | 12 | 16.4 | 0.710 |
| 7 | 21 | 18 | 20 | 20 | 20.0 | 0.713 |
| 8 | 26 | 20 | 16 | 20 | 20.5 | 0.712 |
| 9 | 22 | 17 | 18 | 23 | 20.2 | 0.713 |

Interpretation: this was on the STALE memmap (107/28 dims, 4 missing type_eff scalars
silently zero-padded). Reshape underperformed old BC_base's 22-26% ceiling, but the gap
is within 200-game eval noise (±2-3% per bot, ±1.5% on the 4-bot mean). Not a verdict
on the reshape itself — the real test is on the human memmap with full 109/30 features.

### New human memmap — DONE

`data/datasets/human_v8_100k/` — generated via `replay_to_memmap.py` streaming from HF
dataset `jakegrigsby/metamon-raw-replays`, filtered to gen9ou, min_rating=1500.

- **100,000 replays** accepted (from 457,739 streamed; 78% skipped by rating filter)
- **199,919 episodes** (both perspectives via `--log-both=True`)
- **5,084,603 records** (14× the old bot memmap's 361k)
- **move_cont_dim=109, switch_cont_dim=30** — full type-effectiveness features present
  (the old bot memmap is 107/28 with zero-padded type_eff)
- **Size on disk: 104 GB** (post-trim; pre-trim was 163 GB due to preallocation slack)
- **Zero errors, zero validation failures** during 2h5min of streaming

Integrity verified: file sizes match N rows exactly, boundary row unchanged pre/post
trim (sum_abs=92.67), no NaN across 100k random sample, every legal-mask row has ≥1
active action, episode_index end row matches num_records exactly.

### Pipeline bug fixes (commit `61c665f`)

Two bugs blocked the regen and forced a manual fix mid-session:

1. `replay_parser.py:37` imported `make_obs_mask_and_slots` — a function removed in the
   Session 34 refactor. Top-level import failed, so `replay_to_memmap.py` couldn't even
   start. Fix: dropped the import (function was only used by the legacy `_parse_perspective`
   path, not called externally).

2. `MemmapV8Writer.finalize()` claimed to "trim memmaps to actual size" but only wrote
   `metadata.num_records` — the raw `.npy` files stayed at `max_rows`, wasting ~60 GB of
   preallocation. Fix: added `_trim_files_to_n_rows()` module-level helper that
   `os.truncate()`s each raw-memmap file to `N * per_row_bytes`, and called it from
   `finalize()` after releasing memmap handles (`mm._mmap.close()` + `gc.collect()`).

The fix was verified on the live data: reclaimed 63.67 GB, no integrity regression.

### Crashes earlier in session (context)
**Three BSODs on 2026-04-23 (all bugcheck `0x0000019C` KERNEL_AUTO_BOOST_INVALID_LOCK_RELEASE, Arg1=0x50):**
- 02:26 AM (killed BC training mid-epoch-2)
- 08:36 AM (after user rebooted)
- 11:12 AM (dump saved to `C:\Windows\MEMORY.DMP`)

Same bugcheck + same Arg1 three times = specific driver synchronization bug. **Not** GPU,
**not** fp16, **not** training code. This is the same cluster pattern flagged in
"Known machine-level issues" below and in earlier Session 31/35 BSOD notes.

**Remediation applied this session:**
1. `model.py` got a small AMP dtype fix (scatter sources cast to dest dtype — commit `cf38cf2`).
   Harmless/good regardless of the crashes; committed before investigation.
2. **NVIDIA driver updated** from 551.23 (Jan 2024, 14 months old) → **Studio 595.79**
   (Mar 10, 2026). Clean install via the installer's "Custom → Perform a clean installation".
   Verify post-reboot with `nvidia-smi` — driver_version should read `595.79`.

**Still NOT done (user deferred — flag in next session):**
- MSI bloatware is still running and is the prime suspect per `NEXT_SESSION.md` machine-
  level warning. Active services on this box right now:
  `MSI Foundation Service`, `MSI NBFoundation Service`, `MSI_Central_Service`,
  `MSI_Companion_Service`, `MSI_VoiceControl_Service`, `NahimicService`.
  If crashes resume after the driver update, uninstall MSI Center / Dragon Center /
  NBFoundation / Nahimic next.
- VirtualBox `VBoxNetLwf` errors on every boot (uninstall VirtualBox if not needed).

### Next concrete step — BC reshape on HUMAN data
1. `nvidia-smi` to confirm driver 595.79 and no lingering processes.
2. Start battle servers (commands in "Quick-reference commands" below).
3. Launch BC reshape training on the new human memmap:
   ```bash
   cd pokemon-ai-starter/pokemon-ai/src
   python -u train_bc.py \
     --memmap-dir data/datasets/human_v8_100k \
     --device cuda --fp16 \
     --d-spatial 256 --d-temporal 512 \
     --n-spatial-layers 3 --n-temporal-layers 3 \
     --n-summary-tokens 4 --dropout 0.05 \
     --lr 1e-4 --weight-decay 1e-4 --grad-clip 2.0 \
     --batch-size 16 --epochs 10 --sched cosine --warmup-steps 200 \
     --eval-games 200 2>&1 | tee bc_reshape_human.log
   ```
4. If another BSOD hits, stop, uninstall MSI bloatware (list below), reboot, retry.

**Expected runtime**: 14× more records than the bot memmap → ~14× epoch time, so ~3 hours
per epoch vs. the old ~13 min/epoch → **~30 hours total for 10 epochs**. Longer than one
session. If time-constrained, start with `--epochs 3` to see initial trajectory, then
continue with `--resume` after reviewing epoch-0/1/2 smart_avg.

### Uncommitted files in the tree (as of this session end)
- `watch.ps1` (project root) — PowerShell system monitor (GPU/VRAM/CPU/RAM/disk every 5s
  to `system_watch.log`). Created this session to catch the next BSOD; not committed yet.
  Safe to keep or commit as-is.

### Commits this session
- `cf38cf2` model.py: cast scatter sources to destination dtype in forward
- `61c665f` Fix replay_to_memmap regen: drop stale import, actually trim files

---

## Current position

**Best checkpoint: `selfplay_v9_20260413_061236/snapshot_2979.pt` at Elo 1058**
(confirmed by `data/eval/elo_exp5_FINAL.json`, 62 players, 1891 matches).

- +38 Elo above strongest heuristic bot (Tactical=1019)
- +30 Elo above previous session-top (sp1984=1028)
- +241 Elo above BC base (817)

Near-tied: `snapshot_2999.pt` (Elo 1055, CI overlaps). Top 7 snapshots are all 1048-1058.

**The architectural ceiling is CONFIRMED at ~1058 Elo.** Exp 5 (200 iters with all
safeguards: adaptive entropy + early stop + EMA PFSP + 400 games/iter) produced no
checkpoint that exceeded sp2979. The plateau is real, measured across 62 players and
1891 matchups. Further gains require architectural changes.

**Team selection test in progress:** testing sp2979 with each of 70 handcrafted OU teams
individually against bots (14,000 games). Early results show savg 51-77% depending
on team — massive variance. Fixed-team play is MUCH stronger than random-from-pool.
Results will be in `team_selection_results.json` when complete.

**Session 37 status — TWO major prep milestones done:**

1. **Metamon study complete** (`docs/METAMON_LEARNINGS.md`). Confirmed from direct config
   read that every Metamon size variant uses 5-8× temporal:spatial d_model — strongest
   evidence yet that Option A (capacity reallocation) is the right next architectural lever.

2. **Multi-gen prep DONE** (commit `18e965c`). Previous sessions had already implemented
   90%: vocab loops gens 1-9, features has Mega/Z-move/Dynamax/Tera flags, team_generator
   has per-gen ban lists, format_from_str parses gen. Session 37 fixed two real bugs —
   `ProceduralTeambuilder` wasn't passing `gen` to `load_pokemon_pool` (always loaded
   gen9 tiers), and OU rating threshold was hardcoded `1695` (gen9 only; gen6-8 use `1760`
   in 2024-04 data). Verified gen6/7/8/9 teams now generate era-appropriate mons
   (gen6 HP Ice Thundurus, gen7 Z-crystal Dragonite, gen8 Double Iron Bash Melmetal).

3. **Capacity reshape implemented** (commit `76acca8`). Added `d_spatial`, `d_temporal`,
   `n_summary_tokens` to PokeTransformerConfig. Legacy defaults (None/0) preserve exact
   sp2979 param count (13,382,572, strict=True load verified). New reshape config
   (256/512/3L/3L/K=4/dropout=0.05) instantiates at 14.28M, forward + multi-turn works
   cleanly, FP16 smoke-tested on CUDA. See `docs/METAMON_LEARNINGS.md` §5 for design rationale.

**Next concrete step:** Launch BC retrain with reshape config. Command below.

---

## Quick-reference commands

### Resume PPO from current peak (sp_0219, the all-time-best smart_avg)

```bash
cd pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/bc/v8_bc_20260423_195603/best.pt \
  --resume data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0219.pt \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 200 --n-iters 200 \
  --warmup-iters 0 \
  --lr 3e-5 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --adaptive-entropy --win-rate-mode ema \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  2>&1 | tee -a ppo_human_bc_lr3e5.log
```

**Critical: `--lr 3e-5` not `1e-4`.** From a 45% BC, lr=1e-4 caused 80%+
KL discard and the policy diverged in 3 iters. lr=3e-5 produces clean
multi-epoch updates.

### BC reshape on human data — done, checkpoint locked

Run dir: `data/models/bc/v8_bc_20260423_195603/`. `best.pt` is epoch 2
weights at smart_avg 45.1% (14.28M params, 256/512 spatial/temporal,
3L/3L, K=4). Use this as `--init-from` for PPO. Per-epoch curve:

| Epoch | smart_avg | val_acc |
|---|---|---|
| 0 | 31.1 | 0.671 |
| 1 | 44.4 | 0.683 |
| **2** | **45.1** | **0.706** ← best.pt |
| 3 | 39.9 | 0.706 |
| 4-9 | 16-22 | 0.71-0.713 |

Classic BC overfit after epoch 2 (val_acc keeps climbing, smart_avg
plateaus then regresses). The reshape + human data + restored type_eff
combo broke the old 22-26% BC ceiling decisively.

### Old BC reshape run on bot data — reference only

Run dir: `data/models/bc/v8_bc_20260423_124909/`. `best.pt` at smart_avg
21.2%. Trained on stale `memmap_v8` (107/28 zero-padded). Don't use as
go-forward base.

### Start 3 battle servers (Windows)
```bash
NODE=/c/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe
BS=/c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js
$NODE "$BS" --port 9000 &
$NODE "$BS" --port 9001 &
$NODE "$BS" --port 9002 &
# Verify:
curl -s -o /dev/null -w "9000:%{http_code} " http://127.0.0.1:9000/action.php?
curl -s -o /dev/null -w "9001:%{http_code} " http://127.0.0.1:9001/action.php?
curl -s -o /dev/null -w "9002:%{http_code}\n" http://127.0.0.1:9002/action.php?
```

### Incremental Elo measurement — the "Useful" set we agreed on (Session 39)

5 snapshots: new BC + early/mid/peak/final from this run. Anchors on
sp2979 (Elo 1058) which is still on disk. Note that ~38 of the 52 old
snapshots in `elo_exp5_FINAL.json` were pruned and won't have real
matches against ours — but sp2979, sp2999, sp0284, sp0839, sp1199, etc.
do exist, so the comparison is meaningful.

```bash
cd pokemon-ai-starter/pokemon-ai/src

# Single shard (~2.5 hr for 5 snapshots × ~12 surviving opponents × ~30 min)
python -u eval_elo_ladder.py --add-to data/eval/elo_exp5_FINAL.json \
  --snapshots \
    data/models/bc/v8_bc_20260423_195603/best.pt \
    data/models/rl_v9/selfplay_v9_20260424_213428/snapshot_0029.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0099.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0159.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0219.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
  --names new_bc sp_pre_29 sp_0099 sp_0159 sp_0219 sp_0229 \
  --n-games 100 --concurrency 70 --device cuda \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --out-json data/eval/elo_session39.json
```

What we expect to find: sp_0219 should land near or above sp2979 (1058)
in Elo terms — internal PFSP wr says we beat sp2979 63%. If Elo confirms,
real strength gain is real. If sp_0219 sits below 1058, the smart_avg
gain is mostly bot-eval-specific noise.

For 3-shard parallel (faster), use `--shard 0/3` etc. on different
servers as in the old eval_elo_ladder.py docs.

### Monitor training (PowerShell)
```powershell
Get-Content "path\to\log.log" -Wait | Select-String "\] Iter|EVAL:|Snapshot|KL early|FATAL|ERROR|PFSP|EARLY STOP|ENT"
```

---

## What's been tried and what works (consolidated)

### What's validated — don't change these
- **BC → PPO pipeline** (standard, works)
- **Terminal-only reward** (tried shaping, always hurt)
- **Entity tokenization** (the architectural breakthrough — preserved at any spatial dim)
- **Distributional value head** (51-bin categorical, not scalar)
- **Temporal model for OU** (needed for 30-60 turn games)
- **gamma=0.9999, clip=0.2, lam=0.95** (PPO standards, confirmed)
- **lam=0.95** (not 0.75 — that was our outlier, fixed in Exp 1 for +25 Elo)

### What's been tested and learned

| Experiment | Change | Result | Conclusion |
|-----------|--------|--------|------------|
| Lambda fix | 0.75 → 0.95 | +25 Elo | **Keep.** |
| Entropy tuning | ent 0.04 → 0.02 → 0.03 → 0.02 | 0.02 from healthy base is sweet spot | **Keep 0.02.** |
| Slot permutation | bench/move shuffle per turn | Small impact, but helped stability | **Keep.** |
| PFSP (vs uniform) | (1-wr)² weighted | +11 Elo, but 12% stale-rating waste | **Keep. Consider EMA to fix staleness.** |
| LR refinement | 1e-4 → 3e-5 | Hit peak sp2999 (Elo 1055), then collapsed | **Lower LR is risky without safeguards.** |
| Games/iter 400 | paired with low LR | More data per update | Secondary — paired with LR choice. |
| **Exp 5 safeguards** | adaptive ent + early stop + EMA | 200 iters, savg=55.3% mean, no collapse | **Ceiling confirmed at ~1058. Safeguards work.** |

### What causes collapse (learned the hard way)

Exp 4 collapsed from Elo 1055 (sp2999) to 785 (sp3054) in 55 iters. Root causes:

1. **Weak ent_coef** (0.02) + consistent "improvement" gradients → slow policy sharpening
2. **Stale PFSP cumulative ratings** → 12% of training on opponents we'd already mastered →
   reinforces narrow strategies → over-specialization
3. **Entropy dropped below 0.65 without triggering safeguards** (old adaptive threshold was 0.55)

This is now fixed — see safeguards below.

### Safeguards available (Session 36 additions)

All configurable, defaults preserve old behavior. Enable explicitly for safety.

**Adaptive entropy (`--adaptive-entropy`):**
Auto-raises ent_coef when entropy drops, auto-lowers when too exploratory. Defaults:
- `--adaptive-entropy-low 0.65` (raise ent_coef below this; was 0.55)
- `--adaptive-entropy-high 0.95` (lower ent_coef above this)
- `--adaptive-entropy-step 0.1` (±10% per iter; was ±5%)
- `--adaptive-entropy-max 0.08` (cap)
- `--adaptive-entropy-min 0.01` (floor)

Would have caught Exp 4 collapse at iter 3020 with ent_coef rising to ~0.05.

**Composite early stopping (`--early-stop`):**
Stops training if `patience` (default 3) consecutive RAW evals show BOTH savg regression
AND 3+ of 4 bots regressing, OR savg drops >2× threshold (handles specialization).
- Based on rolling-3 "best" baseline, raw-eval comparison for current
- Would have stopped Exp 4 at iter 3055 with only ~3 Elo loss from peak

Parameters: `--early-stop-patience 3`, `--early-stop-savg-threshold 2.0`,
`--early-stop-bot-threshold 3.0`, `--early-stop-bot-count 3`, `--early-stop-min-evals 5`.

9 unit tests in `test_early_stop.py` validate all trigger conditions including Exp 4 replay.

**EMA PFSP (`--win-rate-mode ema`):**
Replaces cumulative win-rate tracking with exponential moving average. Fixes the 12%
stale-oversampling problem. Old cumulative data naturally fades; current policy strength
reflected in rating.
- `--win-rate-ema-alpha 0.3` (blend weight for new batches)
- `--win-rate-ema-window 50` (cap on effective_games to prevent unbounded growth)

Default `cumulative` preserves exact old behavior. 8 unit tests in `test_ema_pfsp.py`.

---

## Code architecture — key files

### Training pipeline
```
train_rl.py           Main training loop + argparse. Calls collect → PPO → eval → save
 ├─ rl_collection.py    collect_v9 async collection + BackgroundCollector (pipeline)
 │   └─ pfsp_sample()   Prioritized opponent sampling — cumulative or EMA
 ├─ rl_player.py        V9RLPlayer + SelfPlayOpponent (shared build_turn_batch)
 ├─ rl_pipeline.py      Multiprocess variants (InferenceServer, MPRLPlayer)
 ├─ inference_batcher.py Async batched GPU forward — key for concurrency speedup
 ├─ ppo.py              Trajectory, GAE, ppo_update, save_checkpoint (atomic writes)
 ├─ model.py            PokeTransformer 13.38M: spatial 384d/4L + temporal 384d/2L
 └─ features.py         Entity tokenization, build_turn_batch, slot permutation
```

### Evaluation / analysis
```
eval_elo_ladder.py     Full Bradley-Terry ladder + --add-to incremental + --shard parallel
battle_agent.py        Inference player (eval, no training augmentation)
analyze_experiments.py Cross-experiment comparison (per-bot stats, correlations)
analyze_pfsp.py        PFSP opponent tracking analysis (staleness, concentration)
analyze_pfsp_timing.py Opponent rotation / encounter timing analysis
analyze_elo_trajectory.py  Elo plot + era aggregates from ladder JSON + eras.json
build_registry.py      Rebuild persistent registry from scratch
```

### Data / registry
```
data/eval/registry/
├── runs.jsonl         Training run configs (auto-logged by train_rl.py)
├── evals.jsonl        Per-iter bot eval results (auto-logged every 20 iters)
└── elos.jsonl         Per-snapshot Elo measurements (auto-logged by eval_elo_ladder.py)

data/eval/
├── elo_exp5_FINAL.json    Canonical Elo ladder (55 players)
├── eras.json                     Training era definitions for trajectory plots
└── trajectory_session36.png      Latest trajectory plot
```

### Tests
```
test_exp2_exp3.py      Slot permutation + PFSP correctness (14 tests)
test_early_stop.py     Composite early stopping trigger logic (9 tests)
test_ema_pfsp.py       EMA win-rate tracking (8 tests)
```

---

## What to do next — MULTI-GEN IS THE PRIORITY

The hyperparameter refinement lever is exhausted (confirmed by Exp 5). Multi-gen
prep comes FIRST before any model size changes, so all future work builds on a
multi-gen-ready foundation.

### IMMEDIATE NEXT: Study Metamon architecture + apply learnings

**PokeAgent ladder results (Session 36):**
Our model (13.38M) submitted to the PokeAgent Challenge live ladder.
- **Skill Rating 1444±30 with TEAM_T** — rank #12, ABOVE MM-Minikazam (4.7M, 1429)
- **Skill Rating 1376±24 with TEAM_AU** — rank #15
- Team choice moved us 68 points (more than all hyperparameter experiments combined)
- Gap to MM-SmallRLG9 (15M, same size as us): 73 points — they use params more efficiently
- Gap to MM-Kakuna (142M, best): 409 points
- We beat ALL heuristic baselines (BH tier: 1185-1240) by 200+ points

**Key insight: Metamon's 4.7M model nearly matches our 13.38M.** The gap is architecture
and methodology, not model size. Metamon is open source — study their design, apply to ours.

**Metamon reference repo cloned:** `metamon_ref/` (shallow clone, read-only reference).
Study these files (DO NOT modify our code to copy theirs — learn principles, apply to our design):
- Architecture: temporal:spatial ratio, entity token handling, layer configs
- BC data pipeline: preprocessing, quality filters, data scale
- Offline RL recipe: Binary+MaxQ methodology (different from our PPO)
- Minikazam (4.7M) vs Kakuna (142M): what scales and what doesn't

**After understanding Metamon's design, THEN do:**

### Multi-gen + gen-agnostic prep

**Why first:** User direction (Session 33, reconfirmed Session 36): multi-gen before
BC scaling or capacity reallocation. Reasoning: if we change model size or retrain BC
BEFORE multi-gen, we'd redo all that work when we add multi-gen later. Do it once.

**What needs to change (the architecture is ALREADY gen-agnostic):**
- `vocab.py`: verify/expand species/move/ability/item tables for gens 6-9
- `features.py`: add gen-specific volatile effects (Mega Evolution for gen6, Z-moves for gen7)
- `team_generator.py`: per-gen team generation + ban lists (currently gen9 only)
- `format_config.py`: add gen6/7/8 FormatConfig entries (n_actions, team_size, etc.)
- `replay_to_memmap_v8.py`: gen filter for scraping gen6/7/8 OU replays
- `battle_server.js`: verify it handles gen6/7/8 formats

**What does NOT change:** model.py, ppo.py, train_rl.py, rl_collection.py, rl_player.py
(all already parameterized via FormatConfig + --format flag).

**Gen scope:** Start with gens 6-9 (fully compatible features). Gens 4-5 need minor
adjustments. Skip gens 1-3 for now (missing abilities/items, different mechanics).

**Estimated effort:** 1-2 sessions for vocab + features + team gen. Multi-gen replay
scraping can run in background.

**Validation:** Train a 1-iter BC sanity check with expanded vocab on existing gen9
data → confirm no breakage. Then scrape gen6-8 replays and train multi-gen BC.

### AFTER multi-gen: architectural experiments

### Option A — Capacity reallocation (Exp 5 from old plan)

**What:** shrink spatial encoder, grow temporal encoder. Current ratio 1:1 (384d/4L spatial,
384d/2L temporal). Metamon uses 5-8:1 temporal:spatial across all model sizes. Proposed:
spatial 256d/3L, temporal 512d/3L (same param count, redistributed).

**Why:** Our temporal model is undersized for OU's 30-60 turn games with heavy hidden info.
The inverted ratio is our biggest architectural anomaly vs published systems.

**Cost:** Requires BC retrain from scratch (~1-2 days compute) + PPO from new BC (~1 week).

**Decision criterion:** If new model hits Elo 1100+ on existing ladder, ratio matters.
If flat at ~1058, capacity allocation isn't the bottleneck.

**Implementation:** change config values in `model.py` (d_model and n_spatial/temporal_layers).
The architecture supports any combination. Main work is BC retrain, not code.

### Option B — BC scaling (multi-gen path)

**What:** scale BC base from 13.4M to 30M+ params. Train on multi-gen data (gens 6-9 OU, not
just gen9). Per `memory/project_multigen_plan.md` this is mostly data-pipeline work since
architecture is already gen-agnostic.

**Why:** Our BC_base is Elo 817. Metamon's base is 1500+. PPO can only climb from where BC
starts — if BC ceiling is low, PPO ceiling is bounded. Bigger BC + more diverse data
= higher starting floor.

**Cost:** Several weeks. Multi-gen vocab prep (1-2 weeks), multi-gen replay scrape (1-2
weeks, mostly automated), 30M BC training (3-7 days), PPO from new base (1-2 weeks).

**Local limit:** 30M is realistic max on 6GB VRAM. For 50M+, need cloud.

**Decision criterion:** 30M BC at Elo 900+ → scaling helps → continue. 30M BC at ~810 → pivot.

**Steps in detail:**
1. Multi-gen vocab expansion (`vocab.py`, `features.py`)
2. Multi-gen replay memmap (`replay_to_memmap_v8.py` + gen filter)
3. 30M BC training (existing `train_bc.py` with larger config)
4. PPO from new base (same recipe, different checkpoint)

### Option C — Search at inference (MCTS)

**What:** add MCTS on top of learned policy/value. Use NN as prior/evaluator,
search for better moves. MIT thesis finding: "RL alone plateaus but RL+search breaks
through." Foul Play (PokeAgent Challenge co-winner) uses pure MCTS with no NN.

**Why:** Our policy has hit its representational ceiling. Search can compensate by
computing forward through difficult positions instead of relying on amortized NN output.

**Cost:** ~1-2 weeks implementation. Needs opponent modeling (opponent team is partially
hidden) + forward simulation via battle_server.

**Risk:** Inference latency. PokeChamp's LLM search times out 1/3 of games. Our search
would be faster (NN forward vs LLM call) but still adds per-move overhead.

**Expected gain:** MIT thesis achieved 1693 Elo with PPO+MCTS. Gen9 OU is harder than
Gen4 Random, but +200-400 Elo over the current policy is plausible.

### Option D — PokeAgent Challenge submission (READY TO GO)

**What:** Submit our model to the PokeAgent Challenge live benchmark. Script written
(`pokeagent_submit.py`), top teams identified (TEAM_AU at 78.5% savg).

**Why:** Ground-truth Elo against Metamon, Foul Play, and other agents. Our internal
"1058" is bot-anchored; external Elo could be very different.

**How to submit (exact steps):**

1. Go to https://battling.pokeagentchallenge.com
2. Click Login -> Create Team (creates an organization account)
3. Open "My Team" -> create a named AI agent (gets username + password credentials)
4. Run:
```bash
cd pokemon-ai-starter/pokemon-ai/src
python pokeagent_submit.py \
    --checkpoint data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
    --username <agent_username_from_step_3> \
    --password <agent_password_from_step_3> \
    --team TEAM_AU \
    --n-games 50 \
    --device cuda
```
5. Check rating at https://battling.pokeagentchallenge.com/ladder

Server: `wss://battling.pokeagentchallenge.com/showdown/websocket`
Discord for support: https://discord.gg/E2DuX5FWF7
Formats supported: Gen9 OU (our format), Gen1-4 OU, Gen9 VGC

**Cost:** ~1-2 hours for 50 games. No code changes needed. Uses existing BattleAgent.
Top team (TEAM_AU) scores 78.5% vs smart bots locally.

**Team selection results** in `team_selection_results.json`. Top 5: TEAM_AU (78.5%),
TEAM_T (77.5%), TEAM_G (77.0%), TEAM_B (75.0%), TEAM_C (73.5%). Use `--team random`
to rotate among top 10.

### Decision recommendation

**If you want decisive answers fast:** do D (PokeAgent submission) first — gives us an
external benchmark in days, informs A/B/C choice.

**If you want to push forward:** do A (capacity reallocation) before B (BC scaling).
A is cheaper (no new data) and tests the architectural hypothesis directly.

**If you want the highest ceiling:** do C (search). Highest-variance outcome but best
potential gain.

**For cloud deployment:** B + C + D path together. Scale up BC, add search, submit to
benchmark. Requires significant compute budget.

---

## Research context

See `docs/RESEARCH.md` §0 for full comparison tables. Key landscape facts (as of Apr 2026):

| System | Format | Elo | Model | Notes |
|--------|--------|-----|-------|-------|
| **Ours (sp2979)** | Gen9 OU | 1058 internal (bot-anchored) | 13.38M | Current peak |
| ps-ppo | Gen9 Random | 1900+ public ladder | ~15M | Random is easier than OU |
| VGC-Bench BCFP | VGC Doubles | 1768 internal | ~10M | Different format/scale |
| Metamon (Gen1OU) | Gen1 OU | 1500+ public | 15-200M | Published best for public ladder |
| Metamon Kakuna | Gen9OU | Unknown | 57M | Post-paper, no public Elo |
| PokeChamp | Gen9 OU | 1300-1500 | GPT-4 (no training) | LLM minimax search |
| PokeAgent PA-Agent | Gen9 OU | Unknown | RL-based | Co-won PokeAgent Track 1 |
| Foul Play | Multi-gen OU | Unknown | No NN (pure MCTS) | Co-won PokeAgent Track 1 |

**No pure RL agent has demonstrated >1800 Gen9 OU on public ladder** as of Apr 2026.
Top human Gen9 OU peaks: 2030-2115 Elo.

**Validated techniques from literature:**
- Entity tokenization (ps-ppo, VGC-Bench) — we have this
- Distributional value head (ps-ppo, Metamon via two-hot) — we have this
- Transformer with long context (Metamon, VGC-Bench) — we have this
- BC → PPO pipeline (all systems) — we have this
- PFSP opponent sampling (AlphaStar, VGC-Bench Double Oracle) — we have this
- Adaptive/constant entropy (EPO 2025) — we have this
- Slot permutation augmentation (no Pokemon paper; E2GN2 theory) — we have this
- Search + RL (MIT thesis, Foul Play, PokeChamp) — **we don't have this**

---

## Reference data

### Confirmed Elo (from `elo_exp5_FINAL.json`)

Top 10 snapshots:
```
 #1  sp2979  1058  [1044, 1070]   ← current all-time best
 #2  sp2999  1055  [1040, 1068]   ← tied (CI overlap)
 #3  sp3179  1053  [1039, 1065]
 #4  sp2299  1049  [1036, 1060]
 #5  sp2039  1047  [1034, 1058]
 #6  sp3199  1045  [1031, 1058]
 #7  sp2799  1042  [1027, 1056]
 #8  sp2359  1041  [1029, 1055]
 #9  sp2139  1040  [1026, 1051]
 #10 sp3019  1039  [1024, 1049]
```

Bot anchors:
```
  Tactical        1019  (top bot)
  Strategic       1018
  SmartDmg/SH     1000  (SH is anchor)
  Random          462   (floor)
```

Collapse reference:
```
  sp3054          785   ← Exp 4 post-collapse (below BC_base!)
  BC_base         817
```

### Critical checkpoints

- `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` — BC base (Elo 817), init-from for all PPO
- `selfplay_v9_20260413_061236/snapshot_2979.pt` — current all-time best (Elo 1058)
- `selfplay_v9_20260415_083340/emergency_iter_3055.pt` — Exp 4 collapse (DO NOT USE)

### Data files / registry
- `data/eval/elo_exp5_FINAL.json` — canonical Elo ladder (55 players, ~1400 matches)
- `data/eval/eras.json` — training era definitions (E0-E12)
- `data/eval/registry/runs.jsonl` — all training run configs
- `data/eval/registry/evals.jsonl` — per-iter bot evals (~200+ entries)
- `data/eval/registry/elos.jsonl` — all Elo measurements

### Infrastructure
- **Device:** RTX 3060 Laptop 6GB VRAM, 16GB RAM, Win10, Python 3.11, CUDA 12.1
- **Battle servers:** portable Node 20 at `tools/node-v20.18.1-win-x64/node.exe`
- **Use `127.0.0.1` not `localhost`** (Windows DNS quirk)

### Stale memmaps warning
`human_v8/memmap_v8` have `move_cont_dim=107, switch_cont_dim=28`. Current code expects
109/30. `dataset.py` auto-pads, but **regenerate memmaps before any BC training** to
get full type_eff features. Regeneration via `replay_to_memmap_v8.py`.

---

## Session handover checklist

For starting a new session:

1. **Read this file top-to-bottom.** It's self-contained.
2. **Check git status** (`git log --oneline -10`) for any un-handed-over work.
3. **Check for running processes** (`nvidia-smi | grep python`) — anything still training?
4. **Check latest Elo:** `data/eval/elo_exp5_FINAL.json` → confirm sp2979=1058 is still the best.
5. **If training didn't finish cleanly, check the emergency checkpoint** in the run dir.
6. **Pick a next step** from "What to do next" above based on time budget.
7. **If running any PPO training, enable safeguards**: `--adaptive-entropy --early-stop`.
8. **For cloud runs especially**: also `--win-rate-mode ema` (fixes staleness, saves compute).

### Known machine-level issues
- BSODs observed Mar 14-Apr 12 (multiple). Pattern stopped after Apr 12. If they resume,
  update NVIDIA drivers (nvlddmkm errors in Event Log) and uninstall MSI bloatware
  (Dragon Center/NBFoundation — crashed on reboot previously, implicated in WMI bugchecks).

---

## Memory file index

User-level memories in `memory/`:
- `MEMORY.md` — compact project summary loaded every session
- `feedback_*.md` — decisions/lessons that should persist
- `project_*.md` — project state notes
- `reference_*.md` — device/landscape references
