# docs/history — journal archive

These files are **historical snapshots**. Each was true as of its timestamp and
has not been maintained since. Several contain framings that were later
refuted.

**Do not treat anything here as current.** For current state, read
[`../CURRENT_STATE.md`](../CURRENT_STATE.md) — that is the only document in this
repo that claims to be true now.

## Why this split exists

The project's docs grew as a journal: each session appended a new file or a new
dated section. That works for recording what happened, but it means a reader
has no way to tell which claims still hold. In the 2026-09-20 session, three
separate stale framings were caught in a single conversation — a refuted Elo
ceiling, a "smart_avg is saturating" reading, and a multi-gen scope decision
that had never been checked against data availability.

The fix: one maintained document that is edited in place, and everything else
explicitly marked as history.

## What's in here

- **Session journals**: `STATUS.md`, `NEXT_SESSION.md`, `SESSION51_NOTES.md`
- **Completed phases**: `PHASE1_*`, `PHASE2_LAUNCH_PLAN.md`
- **Superseded versions**: `V7_*`, `V8_*`, `PROJECT_PLAN.md`, `REWRITE_DESIGN.md`, `ARCH_AUDIT.md`
- **Cloud infra arc** (parked with the pods 2026-09-20): `S67_*`, `S68_CLEANUP_DESIGN.md`, `MP_DISK_REDESIGN.md`, `MULTIPROCESS_COLLECTION.md`, `CENTRALIZED_INFERENCE_DESIGN.md`, `CLOUD_DEPLOY.md`, `PPO_CLOUD_COOKBOOK.md`, `PPO_PHASED_TRAINING.md`, `PROFILE_BOTTLENECKS_REPORT.md`, `COST_LEDGER.md`
- **Completed experiments**: `AWR_*`, `FISHBOWL_*`, `REPLAY_REHEARSAL_*`, `EXTERNAL_OPPONENTS_PHASE2.md`
- **Deferred TODOs**: `TODO_MCTS_RUN9.md`, `POKE_ENV_MM_STALL_UPSTREAM_DRAFT.md`

Nothing was deleted. Everything here is recoverable and still tracked in git.
