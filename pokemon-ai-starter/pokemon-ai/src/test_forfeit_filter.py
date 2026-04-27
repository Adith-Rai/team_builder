"""test_forfeit_filter.py — Layer-1 forfeit-finish heuristic tests.

V9RLPlayer._finish_looks_real distinguishes legitimate KO-chain finishes
from abrupt WS-disconnect "wins" that the local battle_server emits when
a subprocess opponent (Foul Play, Metamon) crashes mid-battle. Without the
filter, those finishes contribute (a) 1-3 turn trajectories with spurious
+1 terminal rewards to PPO and (b) inflated W/L counts to PFSP weight
updates. See docs/EXTERNAL_OPPONENTS_PHASE2.md bug #11.

Run as a script: `python test_forfeit_filter.py` — exits non-zero on
failure. No pytest dependency (matches the existing test_*.py style).
"""
from unittest.mock import MagicMock

from rl_player import V9RLPlayer


# ---------------------------------------------------------------------------
# Test scaffolding — V9RLPlayer's __init__ has heavy deps (poke-env Player,
# torch, the inference batcher, etc.); for unit-testing the heuristic we
# construct a minimal stub that exposes only what _finish_looks_real reads.
# ---------------------------------------------------------------------------

class _StubPlayer:
    def __init__(self):
        self._self_forfeited = set()


# Bind the actual method to the stub so we test the SAME code path the
# trainer runs in production. If the heuristic body changes in rl_player.py
# but the test isn't updated, we'd want this test to fail.
_StubPlayer._finish_looks_real = V9RLPlayer._finish_looks_real


def _mk_team(n_mons: int, n_fainted: int) -> dict:
    return {f"p{i}": MagicMock(fainted=(i < n_fainted)) for i in range(n_mons)}


def _mk_battle(my_n=6, my_f=0, opp_n=6, opp_f=0, max_size=6, tag="b"):
    b = MagicMock()
    b.battle_tag = tag
    b.team = _mk_team(my_n, my_f)
    b.opponent_team = _mk_team(opp_n, opp_f)
    b.max_team_size = max_size
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_real_ou_win_kept():
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(opp_f=6)) is True


def test_real_ou_loss_kept():
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(my_f=6)) is True


def test_fp_forfeit_win_flagged():
    """FP crashed mid-battle, server emitted |win|RL with their team alive."""
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(opp_f=2)) is False


def test_turn0_disconnect_flagged():
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(opp_f=0, my_f=0)) is False


def test_tie_via_mutual_ko_kept():
    """Both teams wiped (rare but legitimate)."""
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(my_f=6, opp_f=6)) is True


def test_max_size_none_uses_team_len():
    p = _StubPlayer()
    assert p._finish_looks_real(_mk_battle(opp_f=6, max_size=None)) is True


def test_4_mon_format_real_loss_kept():
    """If a future format uses smaller teams and max_team_size is None."""
    p = _StubPlayer()
    b = _mk_battle(my_n=4, opp_n=4, my_f=4, max_size=None)
    assert p._finish_looks_real(b) is True


def test_self_forfeit_at_turn_cap_kept():
    """V9RLPlayer.choose_move turn-cap path forfeits — that's a real loss
    with real play, not an abrupt WS drop."""
    p = _StubPlayer()
    b = _mk_battle(my_f=2, opp_f=1, tag="turncap-x")
    p._self_forfeited.add("turncap-x")
    assert p._finish_looks_real(b) is True


def test_opp_team_empty_flagged():
    """Disconnect before teampreview ever reached opponent_team."""
    p = _StubPlayer()
    b = MagicMock()
    b.battle_tag = "x"
    b.team = _mk_team(6, 0)
    b.opponent_team = {}
    b.max_team_size = 6
    assert p._finish_looks_real(b) is False


def test_both_teams_empty_no_max_size_flagged():
    p = _StubPlayer()
    b = MagicMock()
    b.battle_tag = "y"
    b.team = {}
    b.opponent_team = {}
    b.max_team_size = None
    assert p._finish_looks_real(b) is False


def test_introspection_error_trusts_finish():
    """If we can't introspect the battle, conservatively keep the finish.
    Better one rare miscount than discarding a real game."""
    p = _StubPlayer()
    class _Boom(Exception):
        pass
    b = MagicMock()
    b.battle_tag = "bb"
    type(b).team = property(lambda s: (_ for _ in ()).throw(_Boom()))
    assert p._finish_looks_real(b) is True


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cases = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in cases:
        try:
            fn()
            print(f"  [ok]   {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  [ERR]  {fn.__name__}: {e!r}")
    print(f"\n{len(cases) - fails}/{len(cases)} pass")
    sys.exit(1 if fails else 0)
