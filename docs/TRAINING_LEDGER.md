# TRAINING LEDGER

What we have tried on the **training/learning** side, what happened, and how
strong the evidence is. Companion to `REFUTED_LOG.md`, which covers the
**infrastructure** side.

This file exists because there was no training-side equivalent, and as a
result the same experiments kept getting half-remembered across sessions —
"didn't adding bots collapse training?" (it never ran), "AWR was a hedge
against the KL gate" (it was redundant with the BC anchor).

> **Maintained in place.** Add rows; edit verdicts when evidence changes.
> Do not append dated sections.
>
> **Labels**: `CONFIRMED` (measured) · `STRONG INFERENCE` · `PROVISIONAL`
> (directionally supported, inside noise) · `OPEN` · `REFUTED` · `NEVER RAN`
>
> ⚠️ **Elo figures below are not comparable across rows.** Bradley-Terry Elo
> re-centres when the pool changes; frozen bots drift ~±50 Elo between
> ladders. Each number is meaningful only against its own ladder. See
> `CURRENT_STATE.md` §5.

Last review: **2026-09-20**

---

## Things that worked

| Change | Result | Label | Era / source |
|---|---|---|---|
| **Type-effectiveness features** | **+222 Elo** (729 → 951) | CONFIRMED | Session 30 · `history/PHASE1_INVESTIGATION_PLAN.md:104-108` |
| PPO on top of the BC base | +241 Elo (817 → 1058) | CONFIRMED | `history/NEXT_SESSION.md:2341` |
| BC v10 transformer + tokenizer rewrite | +16 Elo (1120.4 → 1135.9) | CONFIRMED | Session 49 · 500g ladder |
| λ 0.75 → 0.95 | +25 Elo — **kept** | CONFIRMED | `history/NEXT_SESSION.md:2495` |
| PFSP vs uniform opponent sampling | +11 Elo — **kept** (12% stale-rating waste) | CONFIRMED | `history/NEXT_SESSION.md:2498` |
| Capacity reallocation, spatial → temporal | Shipped: `d_temporal=512` vs `d_model=256` | CONFIRMED | supersedes the S33 memo's 1:1 premise |

### The single most important row, and its caveat

Type effectiveness is the largest feature-engineering gain in project history.
But it landed **at 729 Elo** — on a model that had not learned type matchups at
all. That is a climbing-out-of-incompetence gain, not a reaching-elite gain.

There is no evidence feature engineering pays anything like that near the
current ceiling, and direct evidence against: Minikazam has **no** precomputed
type effectiveness and beats us 84-16 (`CURRENT_STATE.md` §5).

**Do not use the +222 to justify expecting a ceiling break from more feature
work.** The encoding/tokenization arc is justified by multi-gen readiness and
completeness, not by this row.

---

## Things that did not work, or did not work as believed

| Change | Verdict | Label | Evidence |
|---|---|---|---|
| **AWR (BC-wins rehearsal, binary filter mix 0.15)** | **Redundant with the BC anchor** — both pull toward BC. Run #6 (no AWR) beat Run #5 (AWR) by 5-20pp on SP-pool and won 5 of 6 externals | CONFIRMED | ~24 Elo of micro-improvement, no decision-pattern effect |
| Synergistic team sources | Both variants landed **inside** the 70-75 plateau (peaks 72.75 / 75.125) | CONFIRMED | Runs #5/#6, 200 iters each |
| LR sweep — 1e-5, 3e-5, 8e-5, 1e-4 | All plateau at 70-74 smart_avg | CONFIRMED | across dense/terminal, prod/dev, fishbowl v2/resume |
| IQL | Dead end | REFUTED | `memory/feedback_iql_dead_end.md` |
| Freeze-spatial / shared backbone | **Rejected on principle** — permanently caps representation learning. Not a data call | REJECTED | `memory/feedback_dont_propose_principle_violations.md` |
| More self-play diversity (S67-EXT) | Expanded the pool **horizontally**, not vertically. Ceiling did not rise | CONFIRMED | Nash-convergent self-play signature |

---

## Open and unresolved

| Question | State |
|---|---|
| **Run #7 — BC anchor removed: exploration valley or real degradation?** | **CLOSED AS UNRECOVERABLE.** Dropped ~200 Elo, then replay analysis at iter 39→49 showed it abandoning setup-spam for principled play (early-setup −36 to −44%, zero setup moves in the iter-49 loss). The iter-99 decision gate never ran, the resume failed for a *mechanical* reason (`--resume` forks a new run_dir), and **all checkpoints past iter 39 are gone**. Only re-running answers it |
| **Run #9 — heuristic-opponent diversity** | **NEVER RAN.** Designed, coded, launch script written; hung on dev because heuristic bots raised exceptions inside CIS. Root-caused and fixed 2026-06-12/13, then work stopped. **Untested, not refuted.** Hypothesis is Elo-supported: anti-setup bots show a 15-30pp differential between setup-spam and principled policies |
| **Why does Minikazam win?** | Open. Candidates, roughly ordered by support: distillation from Alakazam (strong teacher), ~10x more self-play volume (~5M battles vs our 320-480k per run), offline RL on human replays vs our online PPO, opponent-pool composition |
| `--resume` third issue (PFSP win_rates path-keyed) | Unverified against current code. Two of three issues appear fixed (`d1c15295`, `05a04d54`) |

---

## Standing measurement rules

1. **Never compare Elo across ladders.** Use frozen-opponent H2H win rate for
   capability claims.
2. **smart_avg is decoupled from MM win rate.** The smart_avg-peak snapshot was
   the *worst* of three against LargeRL/MediumRL_Aug. Treat smart_avg as a cheap
   in-run smoke signal only.
3. **Training-time per-opponent WR understates H2H by ~30pp** (sampling with
   entropy vs greedy argmax). Good for direction, useless for level.
4. **n=500/pair gives ±4.5pp.** Single deltas under that are not meaningful;
   only monotonic patterns across several snapshots are signal.
