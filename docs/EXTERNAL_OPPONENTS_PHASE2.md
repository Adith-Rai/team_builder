# External Opponents Integration — VALIDATED END-TO-END

> **STATUS (end of Session 42, 2026-04-26): all four opponent paths play
> a full battle to completion against a poke-env 0.10 sender on the local
> battle_server, validated with `diag_cross_venv.py`:**
>
> | Path | Adapter | Validated |
> |---|---|---|
> | Self-play | `SelfPlayOpponent` (in-process) | ✓ |
> | Foul Play MCTS core | `mcts` / `pokeengine` (in-process via poke-engine) | ✓ |
> | Real Foul Play | `foulplay` subprocess in `foul_play_venv` | ✓ |
> | Metamon | `metamon` subprocess in `metamon_venv` | ✓ (Minikazam) |
>
> The hybrid subprocess design works. The earlier "drop the external user
> design and rewrite as in-process Players" recommendation is **obsolete**
> — the failures we hit in Session 39 were all server-side protocol bugs
> in `battle_server.js`, not architectural limits of the subprocess design.
> Once those bugs were fixed (Session 42), the original design completes
> battles in 1.7s (Metamon) / 19s (Foul Play 100ms MCTS) / 6s (mcts)
> against `diag_cross_venv.py`.
>
> **The four bugs we found and fixed (read these so future sessions don't
> rediscover them):**
>
> 1. **`battle_server.js` /challenge PM had wrong field count** (Session 39,
>    commit `e01a37f`). Foul Play silently rejected our PM until we matched
>    real Showdown's 8-pipe / 9-split-field format.
>
> 2. **Foul Play's `accept_challenge` did exact-string username comparison**
>    (Session 39 patch to `foul_play_ref/fp/websocket_client.py`). Showdown
>    sends usernames in lowercase id-form (`diagsender`); FP compared against
>    the CamelCase `--ps-username` arg (`DiagSender`). Patched to use a local
>    `_to_id` helper. **Patch is on disk in `foul_play_ref/`, not committed
>    upstream.** If `foul_play_ref/` is re-cloned, re-apply.
>
> 3. **`battle_server.js` bundled all init events into one ws frame**
>    (Session 42). poke-env 0.10 wants this; Foul Play's parser does
>    `msg.split("|")[2]` expecting the slot, gets the prior field's value
>    (e.g. a `|t:|<timestamp>` value). And Metamon's parser saw the
>    bundled `|player|...|request|...|player|...` and re-counted pokes
>    from a stale poke_list slot, raising `UnusualTeamSize: 7 pokemon`.
>    Fix: detect the recipient via username and switch to a 5-frame
>    Showdown-faithful layout for FP/MM clients (init+title together,
>    each `|player|` alone, the rest minus `|request|` together,
>    `|request|` standalone with an injected monotonic rqid). poke-env 0.10
>    keeps the bundled layout. See `pumpPlayer` in `battle_server.js`.
>
> 4. **`battle_server.js` only handled `/choose` and `/team`, not bare
>    `/switch` or `/move`** (Session 42). Foul Play's `format_decision`
>    sends `/switch <index>|<rqid>` directly (not via `/choose switch`).
>    Battle hung silently mid-battle the first time a forced switch fired.
>    Fix: handle `/switch ` and `/move ` symmetrically with `/choose`.
>    All four also strip the trailing `|<rqid>` before forwarding to
>    BattleStream — real-Showdown clients always append it; BattleStream
>    doesn't understand it and rejects the choice.
>
> 5. **`isShowdownFaithful('mmminikazam')` returned false** (Session 42).
>    The faithful-detection check looked for `mm-` prefix, but the username
>    arrives in toId form with the dash stripped. Fix: also accept
>    `/^mm[a-z]/` plus a `mm-` check on the display name. False positives
>    are harmless (poke-env 0.10 also accepts the per-event layout — real
>    Showdown sent it that way for years).
>
> 6. **Multi-battle: `/leave <battle-tag>` was a no-op; FP hung after every
>    battle** (Session 42, second pass). FP's `leave_battle` blocks on
>    `while True: msg=recv(); if tag in msg and "deinit" in msg: return`.
>    Real Showdown emits `>battle-tag\n|deinit` after `/leave`; our server
>    silently dropped the `/leave`. So FP completed battle 1, sent /leave,
>    waited forever for the deinit echo. Fix: in `cmdBody.startsWith('/leave ')`
>    branch, parse the trailing tag and emit `>tag\n|deinit` to the user.
>    FP sends /leave as a *global* command (`|/leave battle-tag`, empty
>    room prefix) NOT a per-battle one — so the global handler is what gets
>    hit, not the per-battle `/leave` at the top of `handleMessage`.
>
> 7. **Multi-battle: pending /challenge consumed mid-battle** (Session 42,
>    second pass). When the sender issues `/challenge` while the target is
>    still in a battle, server emits `|pm|/challenge` to the target — but
>    target's `pokemon_battle` loop is the one calling `receive_message`,
>    and it silently swallows /pm (no handler in mid-battle parsing).
>    When `pokemon_battle` returns and target loops back to
>    `accept_challenge`, the /pm is already gone and accept loops forever.
>    Fix: in `cleanupBattle`, after a battle ends, iterate
>    `pendingChallenges` for any targeted at users who just became idle and
>    re-emit `|updatechallenges|...` + `|pm|...`. Idempotent — extra /pms
>    are harmless if the target is already in accept_challenge.

> **Multi-battle proof:** with bugs #6 and #7 applied,
> `diag_cross_venv.py --opponent FoulPlayBot --n-games 2` runs
> back-to-back battles end-to-end (`done. W=2 L=0` in 38.7s, both clean).

> **OPEN — Metamon multi-battle / amago session management.** Metamon's
> amago `evaluate_test` loop wraps poke-env's `openai_api` differently
> from FP's straight `accept_challenges`. When the trainer's send_challenges
> doesn't deliver a challenge in time (e.g. PFSP picked another opponent
> first), amago's `env.reset()` fires `RuntimeError: Agent is not challenging`
> and the metamon subprocess crashes. Single-battle Metamon works (1.7s
> diag); multi-battle inside a real PPO iter still has timing issues with
> back-to-back challenges separated by other opponents. Likely fix is
> either (a) a longer poke-env idle timeout in metamon_accept_serve.py, or
> (b) loosening amago's strict reset-must-be-challenging guard. Defer to
> next session — does not block the FP path or `mcts` in-process path.
>
> **One Windows-specific Metamon gotcha:** `_factory_metamon` in
> `external_adapters.py` now sets `TORCHDYNAMO_DISABLE=1` automatically on
> Windows. Metamon's amago integration tries `torch.compile` on first
> inference; that needs Triton; Triton has no Windows wheels; the agent
> crashes before its first move otherwise.

## Reproducing the smoke (5 min, no code changes needed)

`diag_cross_venv.py` is the canonical end-to-end smoke. It spins up a
minimal poke-env 0.10 client (`_Sender`, random moves) and sends one
challenge to a target username. If the target completes the battle —
`[diag] OK — battles 1 in <T>s` — the bridge works.

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src

# Terminal 1 — battle server (one-shot, kill manually after)
../../../tools/node-v20.18.1-win-x64/node.exe battle_server.js --port 9000

# Terminal 2 — Foul Play subprocess
../../../foul_play_venv/Scripts/python.exe -u foul_play_accept_serve.py \
    --username FoulPlayBot --server-port 9000 \
    --num-battles 1 --search-time-ms 100 \
    --team-queue ../../../data/external_team_queue/foulplay \
    --queue-wait-timeout-s 300 --log-level INFO
# Wait for: "iter 1/1 — got team, awaiting challenge"

# Terminal 3 — enqueue ONE team for FP, then run diag
python -c "
import sys; sys.path.insert(0, '.')
from team_generator import enqueue_team
from teams_ou import random_pool_teambuilder
enqueue_team('../../../data/external_team_queue/foulplay',
             random_pool_teambuilder().yield_team())
"
python -u diag_cross_venv.py --opponent FoulPlayBot --port 9000 --timeout-s 180
# Expect: [diag] OK — battles 1 in 19.4s, FP log: "Winner: FoulPlayBot"

# Same recipe for Metamon (different venv + cache var):
TORCHDYNAMO_DISABLE=1 METAMON_CACHE_DIR=$(pwd)/../../../metamon_cache \
../../../metamon_venv/Scripts/python.exe -u metamon_accept_serve.py \
    --model Minikazam --username MM-Minikazam \
    --server-port 9000 --num-battles 1 --format gen9ou \
    --team-queue ../../../data/external_team_queue/metamon \
    --queue-wait-timeout-s 300
# Then: enqueue → diag --opponent MM-Minikazam → expect 1.7s clean win.
```

**Critical setup detail:** `QueueTeambuilder.__init__(clean_on_init=True)`
deletes any pre-existing files in the queue directory. So you must enqueue
**after** the subprocess has started. In production
(`rl_collection.py:_play_one_opponent`), the coordinator enqueues right
before calling `send_challenges` — that's already wired. For the manual
smoke, just enqueue after the subprocess prints "got team / awaiting".

## Prereqs
- PPO has finished and battle servers are idle (or only running 9001/9002,
  leave 9000 for smoke testing if convenient)
- ~30 GB free disk (estimate — amago + metamon deps + HF model cache + Foul
  Play deps + Metamon team data)
- Python 3.11 (matches our main env)
- Working internet for HF / pip downloads

## 1. Foul Play (~30 min, simplest first) — VALIDATED

Light deps. No torch, no GPU.

**Protocol fixes required (already applied to our codebase + upstream clone):**

1. `pokemon-ai-starter/pokemon-ai/src/battle_server.js`: send the |pm|
   /challenge in Showdown standard format (8 pipes, 9 split-fields). Was
   inline-format which Foul Play silently rejected. Committed as e01a37f.

2. `foul_play_ref/fp/websocket_client.py:accept_challenge`: do case-
   insensitive "id"-form comparison of the target username (Showdown sends
   lowercase ids; --ps-username was CamelCase). Patch (apply to a fresh
   clone — not committed to our repo since foul_play_ref is upstream):

   ```python
   # near line 153 in fp/websocket_client.py
   def _to_id(s):
       return ''.join(c for c in s.lower() if c.isalnum())
   if (
       len(split_msg) == 9
       and split_msg[1] == "pm"
       and _to_id(split_msg[3].replace("!", "").replace("‽", ""))
       == _to_id(self.username)
       and split_msg[4].startswith("/challenge")
       and split_msg[5] == battle_format
   ):
       username = split_msg[2].strip()
   ```

   Without this, Foul Play's accept_challenge loop sits forever even
   after the protocol-format fix.

After these two patches the smoke test below passes.

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder

# Create + activate venv (Windows path; on bash use Scripts/activate)
python -m venv foul_play_venv
source foul_play_venv/Scripts/activate

# Install. poke-engine builds via cargo (~5 min compile).
cd foul_play_ref
pip install -r requirements.txt
cd ..

# Smoke test: launch Foul Play in accept_challenge mode pointed at our local
# server. It should connect, print "Logged in as FoulPlayBot", sit idle.
foul_play_venv/Scripts/python.exe foul_play_ref/run.py \
  --websocket-uri=ws://127.0.0.1:9000/showdown/websocket \
  --ps-username=FoulPlayBot \
  --ps-password=fp-changeme \
  --bot-mode=accept_challenge \
  --pokemon-format=gen9ou \
  --team-name=gen9/ou \
  --run-count=10 \
  --search-time-ms=200 \
  2>&1 | head -50

# In another terminal: send a manual challenge from a poke-env script,
# confirm Foul Play accepts and plays a battle. Use battle_agent.py with
# --opponent-name=FoulPlayBot or a small standalone script.

deactivate
```

If Foul Play doesn't have a default gen9ou team folder, populate
`foul_play_ref/teams/teams/gen9/ou/` with at least one team file (Showdown
export format) before running.

## 2. Metamon (~1-2 hr, heaviest install)

Needs amago + metamon. Both have substantial dependencies (torch,
transformers, etc.). amago is the heaviest.

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder

python -m venv metamon_venv
source metamon_venv/Scripts/activate

# Install amago first per their docs.
# Their install instructions: https://ut-austin-rpl.github.io/amago/installation.html
pip install amago

# Install metamon as editable from our cloned ref.
cd metamon_ref
pip install -e .
cd ..

# Set the cache dir (required by metamon — see pretrained.py:35).
# Pick a dir with 5-20 GB free; downloads accumulate here.
export METAMON_CACHE_DIR=$(pwd)/metamon_cache
mkdir -p "$METAMON_CACHE_DIR"

# Verify imports work and list available models.
metamon_venv/Scripts/python.exe -c "
from metamon.rl.pretrained import get_pretrained_model_names
print('Available Metamon models:')
for n in get_pretrained_model_names():
    print(' ', n)
"
# Expected: ~30 entries including Minikazam, SmallRL, MediumRL, etc.
# Compare against pokemon-ai-starter/pokemon-ai/src/metamon_local.yaml.
# Fix any mismatched names in the YAML.

# Smoke test: launch ONE Metamon variant via their serve_model launcher.
# This uses ladder-queue mode (QueueOnLocalLadder), which is the protocol
# mismatch we noted in metamon_local.yaml. The bot will join the local
# server's ladder queue and wait for an opponent.
metamon_venv/Scripts/python.exe -m metamon.rl.self_play.serve_model \
  --username=mm-minikazam \
  --format=gen9ou \
  --config=pokemon-ai-starter/pokemon-ai/src/metamon_local.yaml \
  --n_challenges=5 \
  2>&1 | head -100

# Expected: bot logs in as mm-minikazam, downloads the Minikazam HF
# checkpoint to METAMON_CACHE_DIR (first time), sits in the ladder queue.
# This first download may take several minutes per model.

deactivate
```

## 3. Resolve protocol mismatch (3-6 hours)

Metamon's `serve_model.py` uses `QueueOnLocalLadder` (joins ladder), Foul
Play uses `accept_challenge` (sits in room). Our PPO uses challenge-mode
via `battle_against`. PFSP weighting requires explicit opponent choice,
which only works in challenge-mode → we need Metamon to accept challenges.

Options, in order of preference:

### 3A. Write a thin Metamon accept_challenge wrapper (~half day)

Replace serve_model.py for our use case. Key pieces:

```python
# pokemon-ai-starter/pokemon-ai/src/metamon_accept_challenge.py
# Run inside metamon_venv: python metamon_accept_challenge.py --model Minikazam ...
import asyncio
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

import metamon
from metamon.rl.pretrained import get_pretrained_model
from metamon.interface import (
    TokenizedObservationSpace, get_observation_space, get_action_space,
    get_reward_function,
)
from metamon.tokenizer import get_tokenizer

class MetamonAcceptChallengePlayer(Player):
    """Wraps a Metamon pretrained model as a poke-env Player in accept_challenge mode."""
    def __init__(self, model_name, *, account_configuration,
                 server_configuration, battle_format='gen9ou', temperature=1.0):
        super().__init__(
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            battle_format=battle_format,
        )
        agent_maker = get_pretrained_model(model_name)
        self.agent = agent_maker.initialize_agent(
            checkpoint=None, log=False, action_temperature=temperature
        )
        # observation/action space setup follows what serve_model.py does
        # ...

    def choose_move(self, battle):
        # Convert poke-env Battle → Metamon obs (TokenizedObservationSpace)
        # Run agent forward → sample action → translate back to poke-env action
        # See metamon_to_amago.py:PSLadderAMAGOWrapper for the conversion logic
        # ...
        return chosen_move
```

Then call `await player.accept_challenges(None, n_challenges=10000)` to
sit idle accepting whatever comes in.

### 3B. Run Metamon in ladder mode but only us in queue at a time

Keep their `serve_model.py` unchanged. From our PPO side, when sampling a
Metamon opponent, also enter ladder queue. The local Showdown server pairs
us with the only other queued participant (the targeted Metamon bot). Risk:
if multiple Metamon bots are in the queue, we get random pairing. Workaround:
keep only one Metamon active at a time (kill/restart between iters), but
that defeats the parallelism we want.

### 3C. Switch entire PPO to ladder mode

Most invasive, breaks PFSP weighting — discarded.

→ Recommend 3A.

## 4. PFSP-pool extension (~3-4 hours, after 3A works)

Make `snapshot_pool` accept external usernames as entries, with a small
`PoolEntry` dataclass:

```python
# rl_collection.py additions
@dataclass
class PoolEntry:
    kind: str  # 'local' | 'external'
    path: Optional[str] = None       # for local: .pt path
    username: Optional[str] = None   # for external: Showdown username
    display_name: str = ''           # for win-rate tracking + iter logs
```

Update `_play_one_opponent` to branch:
- `kind == 'local'` → existing `SelfPlayOpponent` + `battle_against`
- `kind == 'external'` → `await player.send_challenges(username, n_battles)`

Win rate tracking keys on `display_name` either way.

## 5. End-to-end validation (~1 hour)

With external_opponents.yaml set to just `mm-minikazam` + `foulplay`:

```bash
cd pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/bc/v8_bc_20260423_195603/best.pt \
  --resume <path/to/sp_0219.pt> \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 100 --max-concurrent 100 --n-iters 10 \
  --warmup-iters 0 --lr 3e-5 --reward-style terminal --lam 0.95 \
  --ent-coef 0.02 --grad-accum 1 --adaptive-entropy --win-rate-mode ema \
  --procedural-teams ../../raw_data/pokemon_usage/2024-04 \
  --external-opponents external_opponents.yaml \
  2>&1 | tee external_smoke.log
```

Expect: `Snapshot pool: N + 2 checkpoints` in iter 0, with the +2 being
mm-minikazam and foulplay. Win rates against them tracked alongside our
own snapshots. No KL collapse, no OOM.

## 6. Bulk add remaining Metamons (~30 min)

Once one external opponent works end-to-end, uncomment / add the remaining
Metamon variants in metamon_local.yaml + external_opponents.yaml. Each is
just another agent entry + opponent entry. Process manager + PFSP pool
handle them automatically.

## Rollback

If anything goes off the rails: `--external-opponents` flag is off by
default, so just don't pass it. The default code path is unchanged.
