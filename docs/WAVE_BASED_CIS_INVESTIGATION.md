# Wave-Based CIS — Investigation State (Track A)

**Branch**: `perf/multi-process-cis-mps` (shared with Track B for continuity)
**Authored**: S66 (2026-05-18) end of session, surfaced by user's local-vs-CIS observation
**Status**: NEW DIRECTION — emerged after discovering local code's wave pattern. Investigation W1-W3 pending.

Sibling memo: `docs/SHARED_BACKBONE_INVESTIGATION.md` (Track B). Next session evaluates both tracks; this is **not** an exclusive-or — they're complementary at the limit, alternatives at the threshold.

---

## §0. Why this exists

End of S66, user noted that in the pre-CIS local pattern (200g/iter, single process), **pool growth didn't materially affect collect time**. Investigation found the mechanism in `pokemon-ai-starter/pokemon-ai/src/rl_collection.py:476`:

```python
# Process in waves of n_servers (parallel within wave, sequential across waves)
for wave_start in range(0, len(opp_tasks), n_servers):
    wave = opp_tasks[wave_start:wave_start + n_servers]
    # One shared batcher for the wave
    batcher = InferenceBatcher(
        model, device, fp16=fp16,
        min_batch=min(8, conc_per_pair * len(wave)),
        timeout_ms=15,
    )
```

**At any moment only `n_servers` (typically 6-8) opps are active.** Pool growth means MORE WAVES, not more concurrent slots. Pool=100 runs as ~17 sequential waves of 6. Each wave looks like a bounded "pool=6 at saturation."

This is **architecturally different** from current CIS, which holds ALL pool opps as simultaneously-live slots (1 player + N opp slots). The CIS pattern creates per-slot starvation at high pool because per-slot arrival rate drops. The wave pattern bounds active slots so each wave's slots are well-fed.

## §1. The proposal

**Wave-based CIS** — refactor CIS to hold a BOUNDED set of active opp slots (matching the local pattern), cycling through pool waves within an iter:

- At iter start: pool_slot_map allocates first K=6 opps to slots 1..6 (slot 0 = player)
- Workers play their share of games against these K opps
- When games for current wave complete (or per-wave game allocation reached): pause workers, reload slots 1..6 with next K opps, resume
- Repeat until full pool covered
- All under single CIS process, no MPS multi-process, no model architectural change

**Math at pool=15 (within currently-real range, user noted pool can reach 15+ active or 100+ total):**

| Param | Value |
|---|---|
| Wave size (K) | 6 active opps |
| Total games/iter | 1600 |
| Games per opp | 1600/15 ≈ 107 |
| Active games per wave | 6 × 107 = 642 |
| Wave wall (estimated) | 555s × 642/1600 ≈ 222s |
| Number of waves | ceil(15/6) = 3 |
| **Total iter wall** | **3 × 222s ≈ 666s ≈ 11 min** |

Comparison:
- Current CIS at pool=15 (estimated): ~1600-2000s
- Wave-based CIS at pool=15: ~666s ✓ (within user's "flat 10-12 min" acceptance)
- Pool=1 baseline: 555s

So wall grows modestly with pool (~20% at pool=15 vs pool=1) but doesn't balloon. This matches the user's local-pattern observation.

## §2. Why this might work AT OUR PROD SCALE (not just local)

The user's question — "why was local pool-invariant?" — has a partial answer in the wave mechanism, but ALSO a confounding factor: at 200g/iter local scale, inference may not have been the bottleneck (other things like WS/asyncio likely dominated). So pool's effect on inference batching was invisible.

**At our 1600g/iter prod scale, inference IS the bottleneck.** Does the wave mechanism still help?

Argument FOR (wave-based CIS will help at prod):
- Within a wave at prod scale: 6 opps × 107 games each = 642 active games per wave. Total inference arrival = 642 × ~0.3 inf/sec/game = ~193 inf/sec, spread across 7 slots (1 player + 6 opp). Player slot gets ~193/sec, each opp slot gets ~32/sec.
- That's HIGHER per-opp arrival than current pool=15 CIS (~13 inf/sec per slot). Batches fill better.
- Per-fire overhead is the same regardless, but fires per second are fewer → CIS thread less saturated.

Argument AGAINST (might not help as much):
- Per-wave setup/teardown cost: snapshot reload into slots, worker rebalancing.
- Snapshot reload from disk is ~5-15s per slot. With 6 slot reloads between waves, +30-90s per wave transition. At 3 waves = 2 transitions = +60-180s per iter.
- The math above ignored this overhead. Adding it: 666s + 120s ≈ 800s. Still under user's threshold but tighter.

Argument MITIGATION:
- Snapshot reload can be done in BACKGROUND while current wave runs. By time wave N completes, slots are pre-loaded for wave N+1.
- Or use a "double-buffered" slot allocation: K+1 slots, K active + 1 loading.

## §3. Why this is LOWER RISK than shared backbone (Track B)

| Dimension | Track A (wave-based) | Track B (shared backbone) |
|---|---|---|
| Quality risk | **ZERO** (no training change) | Small (freeze spatial — validatable) |
| Model architecture change | None | Module-level (frozen/specialized split) |
| BC v10 compat | Used as-is | Used as-is |
| Snapshot format change | None | Specialized-only state_dict |
| Quality validation needed | No (purely orchestration change) | Yes (~$10-30 pod for A/B) |
| Pool ceiling | Bounded growth (works to any pool, slightly slower per wave) | True pool-invariance (constant) |
| Impl complexity | Medium (refactor CIS + worker reassignment) | High (model refactor + training change + quality experiment + impl) |
| External opp compat | Trivially same as today | Trivially same as today |

**Track A is the lower-risk path that probably suffices.** Track B is the architecturally-purer path that scales to anything.

## §4. Investigation steps for next session

**W1 — Read the local code in depth** (1 session)
- `rl_collection.py` end-to-end, especially the wave loop (line 476+) and `InferenceBatcher` usage at line 480
- `inference_batcher.py` — full mechanism
- `rl_player.py` — how `V9RLPlayer` + `SelfPlayOpponent` interact
- Understand: how does the wave swap opps? What's reloaded? What's preserved?
- Output: notes on the exact mechanism + what would need to translate to CIS

**W2 — Design wave-based CIS** (1 session)
- How to keep player slot persistent across waves (don't reload between waves)
- How to swap opp slots: per-wave reload via `load_state_dict` to slots 1..K
- Worker reassignment: which workers play which opp in current wave
- Snapshot reload background-overlap (key for total wall time)
- pool_slot_map extension: maps to current wave's K active opps, not full pool
- Output: design memo

**W3 — Smoke validation experiment** (1 session, ~$5-10 pod)
- Implement basic wave-based CIS on a branch
- 100g/iter smoke run at pool=10 with wave_size=5
- Validate: pool=10 wall ≈ 2 × pool=5 wall? Or less (with overlap)?
- Track per-wave timing
- If smoke validates the wave mechanism scales as projected: proceed to prod-scale impl

**W4 — Prod-scale validation** (~$20-30 pod)
- Full 1600g/iter run with pool=15
- Gate: collect wall ≤ ~800s (allowing some overhead vs 666s estimate)
- If yes: ship wave-based CIS, defer Track B
- If no: diagnose, may need Track B as supplement

## §5. Open questions

1. **Snapshot reload cost at our scale**: ~5-15s per slot per CIS Phase 4.6 design. Background-loading critical. How does this interact with the existing pool_slot_map orchestration?

2. **Worker pacing across waves**: workers play a finite number of games per wave. Some finish faster than others. How to handle tail?

3. **What's the right wave_size?** Smaller waves = more waves = more transition overhead. Larger waves = more concurrent slots = closer to current CIS behavior. Sweet spot likely 6-8 (matches local's n_servers default).

4. **Does the existing PFSP allocator design assume all-opps-live?** Need to verify pool_slot_map can be re-keyed per wave.

5. **Quality consistency**: workers play different opps in different waves within one iter. Does PPO see this as one consistent batch of trajectories, or does the per-wave swap introduce within-iter heterogeneity that hurts gradient stability? **Likely fine** since trajectories are independent samples, but worth thinking about.

## §6. How Track A and Track B relate

NOT mutually exclusive:
- Ship Track A first: fastest path to acceptable collect wall, no quality risk
- Add Track B later if needed: if pool >> 15 becomes a goal, shared backbone unlocks more
- COMBINED Track A + B: wave-based dispatch of shared-backbone forwards. Caps active concurrent slot count AND amortizes shared work. Best of both worlds.

If only one ships: Track A is the LOWER RISK, FASTER path. Track B is the LONGER-TERM correct path.

## §7. What's NOT to do for either track

Per `docs/REFUTED_LOG.md` and S66 findings:
- Multi-process CIS at N≥3 via MPS (Phase A refuted)
- CPU opp inference (S65 + S66 reconfirmation)
- Triton (gRPC + export)
- BC retrain (user explicit)
- Pool size hard cap (user explicit — pool sizing should be a tunable experiment)
