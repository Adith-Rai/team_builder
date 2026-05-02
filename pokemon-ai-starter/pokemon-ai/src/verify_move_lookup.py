# verify_move_lookup.py
# Verification per REWRITE_DESIGN.md §6.1 + Week 1 sub-task #3 of next-prompt.txt:
# "sample 50 moves at random; for each, build a fake battle with that move on
#  the active Pokemon, run features.py's active path, and assert the 107-dim
#  output matches the lookup table to within fp32 noise."
#
# Approach: re-instantiate poke_env.battle.Move(name, gen=lookup_meta["gen"])
# for 50 random move IDs and re-run features.py:_project_move_flags(move). The
# lookup must match bit-for-bit on the battle-state-INDEPENDENT dims, and may
# diverge ONLY on documented battle-state-dependent dims (STAB, current_pp,
# disabled). We enumerate those and confirm the divergence is only on those
# expected positions.
#
# Run: cd pokemon-ai-starter/pokemon-ai/src && python verify_move_lookup.py

from __future__ import annotations
import random
from pathlib import Path

import torch
from poke_env.battle import Move

from features import _project_move_flags
from vocab import Vocab
from model_transformer import (
    load_move_flag_lookup, MOVE_FLAG_DIM, MOVE_FLAG_DIM_BASE, MOVE_BANK_FIELDS,
    MOVE_FLAG_EXTRA, LOOKUP_SCHEMA_VERSION, _extra_move_flags,
)


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
    blob = load_move_flag_lookup(Path("data/lookup/move_flags_v1.pt"))
    flags = blob["flags"]
    banks = blob["banks"]
    valid = blob["valid"]
    meta = blob.get("meta", {})
    print(f"  schema_version={meta.get('schema_version')}  gen={meta.get('gen')}  "
          f"vocab_n_moves={meta.get('vocab_n_moves')}  flag_dim={meta.get('move_flag_dim')}")
    assert meta.get("schema_version") == LOOKUP_SCHEMA_VERSION
    assert meta.get("move_flag_dim") == MOVE_FLAG_DIM
    assert list(meta.get("bank_fields", [])) == list(MOVE_BANK_FIELDS)
    print(f"  shape: {tuple(flags.shape)}  banks: {tuple(banks.shape)}  "
          f"valid moves: {int(valid.sum().item())}")

    print("Loading vocab...")
    v = Vocab.load()
    id_to_name = v.id_to_name_map("move")

    n_moves = flags.shape[0]
    valid_ids = [i for i in range(n_moves) if bool(valid[i].item()) and i in id_to_name]
    rng = random.Random(0)
    sample = rng.sample(valid_ids, k=min(50, len(valid_ids)))
    print(f"Verifying {len(sample)} sampled moves vs features.py:_project_move_flags...")

    n_pass_strict = 0
    n_pass_with_known_divergence = 0
    failures = []

    gen = int(meta.get("gen", 9))
    for mid in sample:
        name = id_to_name[mid]
        try:
            move = Move(name, gen=gen)
        except Exception as e:
            failures.append((mid, name, f"Move() failed: {e}"))
            continue

        # 1) Strict re-call: first 107 dims must match _project_move_flags;
        # last 12 must match _extra_move_flags (Postscript F).
        d_re = _project_move_flags(move)
        re_cont = torch.tensor(d_re["continuous"], dtype=torch.float32)
        diff_base = (re_cont - flags[mid, :MOVE_FLAG_DIM_BASE]).abs().max().item()
        if diff_base > 1e-6:
            failures.append((mid, name, f"strict diff (first 107) {diff_base:.2e} > 1e-6"))
            continue
        re_extra = torch.tensor(_extra_move_flags(move), dtype=torch.float32)
        diff_extra = (re_extra - flags[mid, MOVE_FLAG_DIM_BASE:]).abs().max().item()
        if diff_extra > 1e-6:
            failures.append((mid, name, f"strict diff (extras 107:119) {diff_extra:.2e} > 1e-6"))
            continue
        # Bank ints in column order from MOVE_BANK_FIELDS.
        re_banks = [int(d_re.get(k, 0)) for k in MOVE_BANK_FIELDS]
        if banks[mid].tolist() != re_banks:
            failures.append((mid, name,
                f"bank mismatch lookup={banks[mid].tolist()} re={re_banks}"))
            continue
        n_pass_strict += 1

        # 2) Active-path simulation: pass poke_types=(move.type,) so STAB=True.
        # Then ONLY index 12 (stab) of the FIRST 107 dims should differ from
        # the lookup. The 12 extras (107..118) are move-static and must be
        # bit-exact regardless. We don't simulate disabled / current_pp here
        # because they require a real battle context.
        try:
            move2 = Move(name, gen=gen)
            d_active = _project_move_flags(move2, poke_types=(move2.type,))
            active_cont = torch.tensor(d_active["continuous"], dtype=torch.float32)
            assert active_cont.shape == (MOVE_FLAG_DIM_BASE,), active_cont.shape
            diff_per_dim = (active_cont - flags[mid, :MOVE_FLAG_DIM_BASE]).abs()
            differing = (diff_per_dim > 1e-6).nonzero(as_tuple=True)[0].tolist()
            allowed = {EXPECTED_DIVERGENCES["stab"]}
            unexpected = set(differing) - allowed
            if unexpected:
                failures.append((mid, name,
                    f"active-path divergence at unexpected indices {sorted(unexpected)} "
                    f"(differing dims = {differing})"))
                continue
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
