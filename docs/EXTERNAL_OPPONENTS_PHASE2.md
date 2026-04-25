# Phase 2 — Install & Smoke-Test External Opponents

> **STATUS (end of Session 39):** the original "spawn external bot as
> separate Showdown client + PPO challenges via `send_challenges`" design
> hit a wall. Server protocol bugs are fixed (committed), Foul Play *does*
> accept challenges and *does* start battles, but the battle never plays
> through to completion. Root cause: `Player.send_challenges()` standalone
> does not coordinate move dispatch the way `Player.battle_against()` does
> when both sides are poke-env Players sharing Python state. With Foul
> Play running as a separate process, there's no shared state, so the
> battle sits idle after init.
>
> **Recommended path going forward:** drop the "external Showdown user"
> design. Instead, write a Python `Player` adapter that runs Foul Play's
> MCTS directly via the `poke-engine` Rust library (which Foul Play itself
> uses). This adapter is a normal poke-env `Player` subclass, so it slots
> straight into the existing PPO collection (`battle_against` works
> as-is, PFSP weighting works as-is). Same pattern for Metamon — write
> a `MetamonPlayer` that wraps their pretrained model + amago agent into
> a `Player` via `choose_move()`. The subprocess + YAML config skeleton
> we built (`external_opponent_manager.py`, `external_opponents_example.yaml`)
> isn't lost work — it's still useful if a bot doesn't have a clean
> Python entry point — but it should not be the default.
>
> Estimated effort for the adapter approach (revised):
> - Foul Play `PokeEnginePlayer`: ~half day. We need to drive poke-engine
>   ourselves (set state, run MCTS, read out best move). poke-engine has
>   a Python interface and Foul Play's own search code (`fp/search/main.py`
>   `find_best_move`) is the working reference.
> - Metamon `MetamonPlayer`: ~half day to a full day. Wrap their amago
>   agent's `step` into `choose_move`. Their `metamon.rl.metamon_to_amago`
>   module is the conversion bridge to study.
> - PFSP pool extension (`PoolEntry` dataclass, branch in
>   `_play_one_opponent`): ~2 hours.
> - End-to-end validation: ~1 hour.
> Total: ~2-3 days for both bots wired in cleanly.

After the running PPO finishes, work through this top-to-bottom. Each step
is small enough to fail fast; if any step breaks, fix before moving on.

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
