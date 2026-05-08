# Cloud cost ledger

Tracks $ spent + $ saved per run. Update at end of each session.

**Cloud rate**: RunPod A100 SXM 80GB at **$1.50/hr**.

---

## Session-level totals (running)

| Session | Cloud hours | Spent | Notes |
|---|---|---|---|
| Session 50 (BC v10 cloud + Phase 1 v3 launch) | ~50 hr | ~$75 | BC v10 e3 ckpt produced (peak Elo 1135.9). Phase 1 v3 launched. |
| Session 50 cont. (mp-disk + leak fix + Phase 1 v3 resumed) | ~10 hr | ~$15 | Smoke + cutover + iters 11-13 |
| Session 51 (Tier 1 + heartbeat fix + CIS Phases 1-4.2 + D1-D3) | ~20 hr | ~$30 | Includes 7+ hour iter 17 hang ($10 wasted on the hung run) + Tier 1 wins |
| Session 52+ projected (Phase 1 v3 compiled run finish + multi-gen prep + BC v11) | ~150-200 hr | ~$225-300 | See breakdown below |

---

## Phase 1 v3 budget (gen 9 OU — V1 baseline)

The full 200-iter Phase 1 v3 cost is the V1 baseline reference.

| Run state | Iters | Hours | Cost |
|---|---|---|---|
| Original (uncompiled, mp-disk) | 11-14 (4 iters at ~75 min) | ~5 hr | ~$7.50 |
| Hung at iter 17 (heartbeat starvation) | 0 (wasted compute) | ~7 hr at 0% CPU | ~$10 wasted |
| Compiled relaunch from snapshot_0014 (this session) | 15-186 (172 iters) | TBD — projecting ~145-170 hr at 50-60 min/iter steady state | ~$220-250 |
| **Total Phase 1 v3 (start to iter 200)** | 200 iters | ~165-185 hr | **~$245-280** |

**Tier 1 savings (compiled vs uncompiled at Phase 1 v3 scale)**:
- Uncompiled was 70-75 min/iter; compiled is 50-55 min/iter
- ~30% per-iter reduction = ~50-65 hr saved over 172 iters
- **~$75-100 saved vs uncompiled run** at this scale

---

## Multi-gen run budget (Phase 3 — gen 6+)

Per `docs/MULTIGEN_FEASIBILITY.md`. Estimates assume Phase 4.3 CIS lands
(another ~30-40% per-iter saving via real `--cis --pipeline` overlap).

| Stage | Estimated wall | Cost |
|---|---|---|
| D4 HuggingFace replay corpus pull (gen 6/7/8) | ~5 hr download + storage | ~$10 (download time) + storage rent ($/GB-month) |
| D5 replay_to_memmap multi-gen (per-gen batch) | ~10 hr cloud parsing | ~$15 |
| D6 BC v11 multi-gen retrain (5 epochs, B=48 fp16) | 5-7 days A100 | ~$180-250 |
| E1 PPO multi-gen run (2-3 weeks at 200-300 iters across 4 gens with CIS) | ~250-400 hr | ~$375-600 |
| **Total multi-gen launch (D4 → first PPO milestone)** | ~3-5 weeks | **~$600-900** |

**Without CIS Phase 4.3**: add ~30-40% to PPO multi-gen cost = ~$120-240 more.

---

## Per-optimization gain reference (Tier 1 + Tier 2)

What each optimization saves over the baseline (validated where shipped):

| Optimization | Per-iter saving | Phase 1 v3 (172 iters left) | Multi-gen 5-7 wks |
|---|---|---|---|
| ✅ torch.compile Path 2 | 30% (measured) | ~$75-100 | ~$300-450 |
| ✅ fused AdamW | 3-7% (Ampere+) | ~$8-20 | ~$30-90 |
| ✅ heartbeat mitigations | not a perf win — prevents catastrophic hang | priceless (avoided $10-50/run hang) | priceless |
| ✅ bf16 (validated, not yet in production) | 0-5% + stability | ~$0-10 | ~$0-30 + cleaner code |
| ⏸ CIS Phase 4.3 (PFSP swap + bg overlap) | additional 30-40% via pipeline overlap | not relevant (Phase 1 done) | ~$200-300 |
| ⏸ Flash attention (10-30% on attention layers) | 5-15% iter-wide | ~$15-40 | ~$60-180 |
| ⏸ adaptive epoch count | 30-40% on first-post-warmup iter | one-time ~$5 | one-time ~$15 per phase warmup |

**Compound expected from current baseline**: if Tier 1 + Phase 4.3 + flash all
land properly, multi-gen run becomes ~50% cheaper than naive `--mp` alone.

---

## Cost-saving rules

1. **Validate small-scale first ($1-3) before launching production ($60-300).**
   Session 50 caught 5 bugs via the 6-test pattern that would have wasted
   40+ hr each (= $60+ each). The cost of NOT validating is much higher.
2. **Don't rerun what's already validated.** `--mp` is production-stable;
   don't re-validate it on every relaunch. Just verify launch banner.
3. **Stop+resume on RunPod is RISKY.** Container Disk re-provisions can
   wipe `/workspace`. Use Network Volume for runs > 1 epoch (see Session
   49 BC v10 incident). Network Volume = $0.07/GB-month, peanuts vs
   compute.
4. **Pod can be terminated automatically if idle.** Don't leave a
   half-finished hung run sitting at 0% CPU for hours. Diagnose + kill
   cleanly. Iter 17 hang cost ~$10 in wasted compute.
5. **Use spot pricing if non-time-critical.** RunPod Spot is ~50% cheaper
   but can be evicted. Good for BC training (resumable from per-epoch
   ckpt), bad for hot PPO production.

---

## How to add to this ledger

End of each session, add a row to "Session-level totals" with:
- Hours of cloud time used
- $ spent
- Major outcomes (what was produced, what was wasted)

If a major optimization shipped, update the per-optimization gain
reference table with measured (not estimated) numbers from production.
