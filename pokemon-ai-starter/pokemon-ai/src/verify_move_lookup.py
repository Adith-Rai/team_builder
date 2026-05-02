# verify_move_lookup.py
# Verification per REWRITE_DESIGN.md §6.1 + Week 1 sub-task #3 of next-prompt.txt:
# "sample 50 moves at random; for each, build a fake battle with that move on
#  the active Pokemon, run features.py's active path, and assert the 107-dim
#  output matches the lookup table to within fp32 noise."
#
# Approach: re-instantiate poke_env.battle.Move(name, gen=9) for 50 random move
# IDs and re-run features.py:_project_move_flags(move). The lookup must match
# bit-for-bit on the 105 battle-state-INDEPENDENT dims, and may diverge ONLY on
# documented battle-state-dependent dims (STAB, current_pp, disabled). We
# enumerate those and confirm the divergence is only on those expected positions.
#
# We also do an "active-path" call that mimics features.py:_encode_action_slots
# by passing poke_types=(move.type,) so STAB=True. The lookup builds with
# poke_types=None (STAB=False), so we expect a difference at exactly the STAB
# index — no other index should differ.
#
# Run: cd pokemon-ai-starter/pokemon-ai/src && python verify_move_lookup.py

from __future__ import annotations
import random
from pathlib import Path

import torch
from poke_env.battle import Move

from features import _project_move_flags
from vocab import Vocab
from model_transformer import load_move_flag_lookup


# Indices in the 107-dim continuous vector that depend on battle state.
# Identified by reading features.py:_project_move_flags (lines 1204-1262):
#   continuous[12] = stab        (depends on user pokemon types)
#   continuous[ 9] = current_pp/64 (depends on remaining pp; lookup uses max pp)
#   continuous[10] = disabled    (depends on whether the move is disabled this turn)
# All other indices depend only on the move's static data.
EXPECTED_DIVERGENCES = {
    "stab":         12,
    "current_pp":    9,
    "disabled":     10,
}


def main():
    print("Loading lookup table...")
    lookup = load_move_flag_lookup(Path("data/lookup/move_flags_v1.pt"))
    flags = lookup["flags"]      # (n_moves, 107)
    banks = lookup["banks"]      # (n_moves, 4)
    valid = lookup["valid"]      # (n_moves,)
    n_moves = flags.shape[0]
    print(f"  shape: {flags.shape}, banks: {banks.shape}, valid moves: {int(valid.sum().item())}")

    print("Loading vocab...")
    v = Vocab.load()
    id_to_name = {idx: name for name, idx in v._move.items()}

    valid_ids = [i for i in range(n_moves) if bool(valid[i].item()) and i in id_to_name]
    rng = random.Random(0)
    sample = rng.sample(valid_ids, k=min(50, len(valid_ids)))
    print(f"Verifying {len(sample)} sampled moves vs features.py:_project_move_flags...")

    n_pass_strict = 0
    n_pass_with_known_divergence = 0
    failures = []

    for mid in sample:
        name = id_to_name[mid]
        try:
            move = Move(name, gen=9)
        except Exception as e:
            failures.append((mid, name, f"Move() failed: {e}"))
            continue

        # 1) Strict re-call without poke_types (matches lookup's build path).
        d_re = _project_move_flags(move)
        re_cont = torch.tensor(d_re["continuous"], dtype=torch.float32)
        diff = (re_cont - flags[mid]).abs().max().item()
        if diff > 1e-6:
            failures.append((mid, name, f"strict diff {diff:.2e} > 1e-6"))
            continue
        # Also assert bank ints match
        if not (banks[mid, 0] == d_re["bp_int"]
                and banks[mid, 1] == d_re["acc_int"]
                and banks[mid, 2] == d_re["pp_int"]
                and banks[mid, 3] == d_re["priority_int"]):
            failures.append((mid, name,
                f"bank mismatch lookup={banks[mid].tolist()} re={[d_re[k] for k in ['bp_int','acc_int','pp_int','priority_int']]}"))
            continue
        n_pass_strict += 1

        # 2) Active-path simulation: pass poke_types=(move.type,) so STAB=True.
        # Then ONLY index 12 (stab) should differ from the lookup. (Plus the 2
        # active-only dims at indices [107, 108] which the active path appends
        # in _encode_action_slots — those aren't part of the 107-dim lookup.)
        # We don't simulate disabled / current_pp here because they require a
        # real battle context; they're documented as static-in-lookup.
        try:
            move2 = Move(name, gen=9)  # fresh instance (pp counters etc.)
            d_active = _project_move_flags(move2, poke_types=(move2.type,))
            active_cont = torch.tensor(d_active["continuous"], dtype=torch.float32)
            assert active_cont.shape == (107,), active_cont.shape
            diff_per_dim = (active_cont - flags[mid]).abs()
            differing = (diff_per_dim > 1e-6).nonzero(as_tuple=True)[0].tolist()
            allowed = {EXPECTED_DIVERGENCES["stab"]}
            unexpected = set(differing) - allowed
            if unexpected:
                failures.append((mid, name,
                    f"active-path divergence at unexpected indices {sorted(unexpected)} "
                    f"(differing dims = {differing})"))
                continue
            # If STAB index was indeed flipped (since move.type is in poke_types),
            # it's expected; if NOT flipped (no STAB-able move type), the lookup
            # already had STAB=0 and active-path also yields 0 — also fine.
            n_pass_with_known_divergence += 1
        except Exception as e:
            failures.append((mid, name, f"active-path call failed: {e}"))

    print(f"\nResults:")
    print(f"  Strict re-call exact match (no battle state):  {n_pass_strict} / {len(sample)}")
    print(f"  Active-path with poke_types (STAB-only diverge): {n_pass_with_known_divergence} / {len(sample)}")
    print(f"  Failures: {len(failures)}")
    for mid, name, msg in failures[:10]:
        print(f"    move_id={mid:4d} name={name!r:24s} {msg}")
    if failures:
        raise SystemExit(1)
    print("\nAll 50 sampled moves match the lookup within fp32 noise.")
    print("Documented battle-state-dependent dims:")
    for k, idx in EXPECTED_DIVERGENCES.items():
        print(f"  cont[{idx:3d}] = {k} (lookup holds the no-battle-state value)")


if __name__ == "__main__":
    main()
