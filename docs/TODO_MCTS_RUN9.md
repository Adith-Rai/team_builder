# MCTS deferred from Run #9 — investigation TODO

**Status**: deferred, not fixed. S68 2026-06-10.

## What happened

When Run #9 (heuristic-pool diversity) was launched on dev pod with the
canonical MCTS setup (mcts-fast + mcts-medium in `--n-ext-per-iter 5`),
worker throughput on MCTS-paired workers became severely slow.

Specific symptoms:
- Workers assigned MCTS opps showed `n_done=0/n` for 5+ minutes while other
  workers progressed normally
- Thousands of `PokeEngine MCTS failed: InvalidWeight (PanicException)`
  warnings in log
- Iter wall time inflated 4× on some iters

Run #7 on prod with the same MCTS adapter setup runs fine — 5,963 MCTS
panics observed across the run, no measurable throughput impact.

## Why prod (Run #7) handles it but dev (Run #9) doesn't

Run #7 vs Run #9 load profile diff:

| | Run #7 prod | Run #9 dev |
|---|---|---|
| Workers | 90 | 70 |
| Games/iter | 1600 | 2240 (+40%) |
| Active opps | 10 | 15 (+50%) |
| **Games per worker per opp** | **~18** | **~30 (+67%)** |
| MCTS panic types | "Encore should not be active" | **"InvalidWeight"** (new) |

The compound effect:
1. **MCTS executor is single-threaded per worker** (`ThreadPoolExecutor max_workers=1`).
   Run #9's 30-deep queue × MCTS latency per battle = workers serialize through
   a tight bottleneck.
2. **New "InvalidWeight" panic type** appears in Run #9 dev environment (not
   seen in Run #7 prod). Per-panic recovery (panic catch + log with traceback
   + smart-fallback call) adds noticeable latency.
3. Compound: 30 battles × frequent panic recoveries × single-threaded
   executor = worker hits effective throughput wall.

Note: the smart-fallback IS implemented and works
(`pokeengine_player.py:486-528` — TacticalPlayer takes over the turn).
The slowness isn't from broken fallback, it's from per-battle latency
increasing under the compound conditions.

## Concrete investigation hooks (before re-adding MCTS)

### 1. Fix or filter the InvalidWeight panic upstream

The panic source is `src/mcts.rs:111:48` in the poke-engine Rust crate
(`called Result::unwrap() on an Err value: InvalidWeight`). Likely:
- A move's calculated weight is 0, NaN, or negative in some battle state
- The Rust crate `unwrap()`s without graceful handling

Two options:
- **a) Patch `_battle_to_pe_state` to filter problematic states** —
  detect what battle conditions trigger InvalidWeight, return None,
  fall back to smart-bot for that turn without invoking MCTS at all.
- **b) Patch poke-engine itself** — replace `unwrap()` with `?` + graceful
  fallback in the Rust crate. Requires building poke-engine from source.

(a) is faster; (b) is the proper fix.

To investigate: log all `_battle_to_pe_state` calls that lead to
InvalidWeight panics. Look for common battle features (specific moves?
specific abilities? specific item states?).

### 2. Bump MCTS executor max_workers from 1 to 2-4

`pokeengine_player.py:437-439`:
```python
self._executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=f"pe-mcts-{id(self):x}"
)
```

The `max_workers=1` was probably chosen to avoid concurrent poke-engine
state mutations. But with newer poke-engine versions, this may not be needed.

Test: change to `max_workers=4`, smoke-test Run #9 setup on dev, verify
no panics related to concurrent state mutation. If clean → ship.

### 3. Scale --mp-workers to opp count

Run #7's 90 workers / 10 opps = 9 workers/opp. Run #9's 70 workers / 15
opps = ~5 workers/opp. The lower workers/opp ratio amplifies per-worker
queue depth (n_for_opp = ~30 vs ~18).

For Run #10+, consider scaling --mp-workers with opp count: aim for
8-10 workers/opp regardless of total opp count. For Run #9-style setup
(15 opps), this would mean 120-150 mp-workers — on dev pod with 80 GB
A100, GPU is fine; CPU and memory needed to be verified.

## When to re-add MCTS

Re-add to Run #10 or later runs WHEN at least one of:
- Hook 1 (filter InvalidWeight upstream) implemented + smoke-tested clean
- Hook 2 (multi-worker MCTS executor) tested clean
- Hook 3 (more --mp-workers) used + verified MCTS throughput acceptable

DO NOT re-add without one of these. Removing MCTS for Run #9 was an
experimental decision (not a fix) — adding it back without addressing
the root cause would just bring back the slowdown.

## Why this isn't a shortcut

Per project principles ("no shortcuts; anything shipped should not need
to be touched again"):
- Removing MCTS from Run #9 is documented + flagged with concrete
  investigation hooks
- Future runs that want MCTS have a clear checklist before re-adding
- The diagnosis (per-worker MCTS queue + new panic type) is preserved in
  this doc + the yaml comment
- The smart-fallback IS verified working — we're not patching around a
  broken fallback, we're side-stepping a latency stack-up

Conversely, Option A from earlier (lower max_concurrent_battles) WOULD
have been a shortcut — that would mask the issue without diagnosis.

## Cross-references

- `pokemon-ai-starter/pokemon-ai/src/pokeengine_player.py:436-468` —
  smart-fallback implementation (TacticalPlayer)
- `pokemon-ai-starter/pokemon-ai/src/pokeengine_player.py:479-528` —
  MCTS choose_move with three-layer panic recovery
- Run #7 prod log: `/tmp/run7_no_anchor_awr_syn_v1.log` (search
  "PokeEngine MCTS failed" — 5,963 instances, runs fine)
- Run #9 dev log: `/tmp/run9_attempt3_multi_instance_BROKEN.log`
  ("InvalidWeight" panic + slow workers)
