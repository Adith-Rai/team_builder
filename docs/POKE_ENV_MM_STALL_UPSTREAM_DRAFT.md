# Upstream issue draft — poke-env / metamon MM hang on "[Invalid choice]"

**Target repos**: hsahovic/poke-env (primary) AND/OR UT-Austin-RPL/metamon (where MM RL bots live)

Copy/paste the relevant content below.

---

## Title

`PSClient hangs indefinitely when "[Invalid choice] Can't switch: You have to pass to a fainted Pokémon" error received`

## Body

### Bug

When the agent policy selects an invalid switch action (e.g., switching to an already-fainted Pokémon during a forced switch), Showdown returns the error message:

```
['', 'error', "[Invalid choice] Can't switch: You have to pass to a fainted Pokémon"]
```

`PSClient._handle_message` logs this as `CRITICAL - Unexpected error message: ...` but doesn't trigger any recovery. The agent's main task then waits indefinitely for a valid action that's never supplied → battle hangs for the full Showdown server timeout (~5+ min observed).

### Concrete data (3 RL-based MM bots, 5-day production sample)

| Bot | "Invalid choice" CRITICAL occurrences |
|---|---|
| MM-LargeRL-{0,1,2} | 2 + 4 + 5 = 11 |
| MM-MediumRL_Aug-{0,1,2} | 3 + 3 + 1 = 7 |
| MM-SyntheticRLV2-{0,1,2} | 3 + 3 + 4 = 10 |
| **Total** | **28** over ~5 days |

Per occurrence: bot hangs ~5 min wall before Showdown server forces battle resolution. In a self-play training context using these bots as opponents, this causes ~30-50 game slots per affected iter to time out.

Older models (Minikazam, SmallRLGen9Beta) hit a different error pattern (ZoroarkException from replay parser) — separate issue.

### Reproduction (rough)

1. Run an RL-based MM bot (e.g., LargeRL) in `AcceptChallengesOnLocal` mode
2. Have the opponent reduce all of MM's Pokemon EXCEPT the active to fainted state
3. KO the MM's active Pokemon
4. MM's policy may select `switch <fainted_index>` → "[Invalid choice]" returns → hang

Exact game-state trigger isn't fully nailed down (the bot picks a switch index that resolves to a fainted mon — could be a teampreview-vs-current-team-state desync, or an inadequate action mask).

### Expected behavior

`PSClient._handle_message` (or higher in metamon's RL agent wrapper) should:
- Recognize the recoverable error class
- Send a default action (e.g., `/choose default`) OR forfeit the battle (`/forfeit`)
- Allow the agent to continue with the next challenge

### Current workaround in our setup

We accept the cost. Our infrastructure:
- Per-iter MM spawn/release (fresh state every iter)
- 300s stall-detection + dispatch cancellation
- Multi-instance MM (3 per model — when 1 hangs, others continue)

These BOUND the damage to ~0.6% game loss / ~5% wall overhead at our scale. But don't PREVENT the trigger.

### Environment

- poke-env: [insert version]
- metamon: [insert version / commit]
- gen 9 OU format
- AcceptChallengesOnLocal mode (single-actor)

---

## After filing

- Link the issue # in `project_s68_mm_stall_open_investigation.md`
- Track for any maintainer response
- If accepted: PR with our recovery logic (would need to draft per-error-class recovery)
- If declined: implement metamon-side wrapper to catch + forfeit (Option A from open investigation memo)
