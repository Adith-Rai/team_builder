# test_tokenizer.py
# Unit + integration tests for model_transformer.Tokenizer (REWRITE_DESIGN.md
# Week 1 + Postscript A/B). Each `test_*` function asserts independently and
# prints a one-line summary. Run as a script (no pytest needed):
#
#   cd pokemon-ai-starter/pokemon-ai/src && python test_tokenizer.py
#
# Test coverage:
#   1. Shape + finiteness on real human_v8_100k data
#   2. Multi-turn forward (3 turns of episode 0)
#   3. Opp Pokemon with unrevealed (move_id=0) moves don't crash
#   4. type_id_embed contributes to output (zero-test)
#   5. pokemon_slot_embed disambiguates same-attribute tokens across slots
#   6. Type-token ordering is canonical (sorted) — no nondeterminism on 2-type Pokemon
#   7. Init weight stats: Linear / Embedding stds ≈ cfg.init_std=0.02
#   8. Gradient flows through opp_threat_unknown
#   9. Active-move real banks override (Postscript B): slot-0 banks differ
#      between active-banks-passed and active-banks-omitted
#  10. Doubles formats explicitly rejected at construction (NotImplementedError)

from __future__ import annotations
from pathlib import Path

import torch

from dataset import MemmapDataset, collate_seq, unpack_turn_batch
from format_config import FormatConfig, FORMAT_SINGLES
from model_transformer import (
    Tokenizer, TransformerConfig, load_move_flag_lookup,
    N_TOKENS, N_TOKEN_TYPES, TT_MOVE,
    MAX_TYPES_PER_POKEMON, _SL_TYPES, _PER_POKEMON_TT, _FIRST_MOVE_OFFSET,
)


MEMMAP_DIR = "data/datasets/human_v8_100k"
LOOKUP_PATH = "data/lookup/move_flags_v1.pt"


def _make_tokenizer(seed: int = 0) -> Tokenizer:
    torch.manual_seed(seed)
    cfg = TransformerConfig.with_vocab_sizes_from_disk()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH), expected_n_moves=cfg.n_moves)
    return Tokenizer(cfg, move_flag_lookup=lookup)


def _load_sample_batch(n_episodes: int = 5):
    ds = MemmapDataset(MEMMAP_DIR, split="train")
    return collate_seq([ds[i] for i in range(n_episodes)])


def _assert_finite(t: torch.Tensor, name: str):
    assert torch.isfinite(t).all(), \
        f"{name} has {(~torch.isfinite(t)).sum().item()} non-finite values"


# ---------------- Real-data smoke tests ----------------

def test_shape_and_finiteness():
    print("== test 1: shape + no NaN/inf on 5 real episodes ==")
    tok = _make_tokenizer().eval()
    collated = _load_sample_batch(5)
    B = collated["our_pokemon_ids"].shape[0]
    print(f"  loaded B={B} episodes, T={collated['mask'].shape[1]} max turns")
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    with torch.no_grad():
        out = tok(batch)
    assert out["tokens"].shape == (B, N_TOKENS, tok.cfg.d_model), out["tokens"].shape
    _assert_finite(out["tokens"], "tokens")
    print(f"  tokens shape: {tuple(out['tokens'].shape)} OK")
    unique_tt = out["type_ids"].unique().tolist()
    assert sorted(unique_tt) == list(range(N_TOKEN_TYPES)), \
        f"type_ids missing some types: {unique_tt}"
    print(f"  all {N_TOKEN_TYPES} token types present: OK")


def test_multiple_turns():
    print("\n== test 2: 3 turns of episode 0 ==")
    tok = _make_tokenizer().eval()
    collated = _load_sample_batch(2)
    seq_lens = collated["seq_lens"].tolist()
    print(f"  episode lengths: {seq_lens}")
    n_turns = min(3, seq_lens[0])
    for t in range(n_turns):
        batch = unpack_turn_batch(collated, t=t, device=torch.device("cpu"))
        with torch.no_grad():
            out = tok(batch)
        assert out["tokens"].shape[1] == N_TOKENS
        _assert_finite(out["tokens"], f"tokens@t={t}")
    print(f"  {n_turns} turns processed without NaN: OK")


def test_unknown_opp_moves():
    print("\n== test 3: opp Pokemon with unrevealed (move_id=0) moves ==")
    tok = _make_tokenizer().eval()
    collated = _load_sample_batch(5)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    opp_move_ids = collated["opp_pokemon_move_ids"][:, 0]
    n_unknown = int((opp_move_ids == 0).sum().item())
    n_total = int(opp_move_ids.numel())
    print(f"  opp move slots @ turn 0: {n_unknown}/{n_total} unknown (id=0)")
    assert n_unknown > 0, "expected some unrevealed opp moves at t=0"
    with torch.no_grad():
        out = tok(batch)
    _assert_finite(out["tokens"], "tokens")
    # Sanity: position-IDs match expected layout for opp Pokemon 0 move 0.
    # Layout: 6 battle-state + 6 our × 17 + opp 0 starts at 108. Move 0 = +13.
    OPP_0_MOVE_0_IDX = 6 + 6 * 17 + _FIRST_MOVE_OFFSET
    assert tok.type_ids[OPP_0_MOVE_0_IDX].item() == TT_MOVE
    assert tok.move_slots[OPP_0_MOVE_0_IDX].item() == 1   # move slot 0 → vocab 1 (0 reserved)
    print(f"  opp_0_move_0 position-IDs OK (idx={OPP_0_MOVE_0_IDX})")


# ---------------- Embedding-contribution tests ----------------

def test_type_id_embedding_contributes():
    print("\n== test 4: type_id_embed contributes to output ==")
    tok = _make_tokenizer().eval()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    with torch.no_grad():
        out_normal = tok(batch)["tokens"]
        saved = tok.type_id_embed.weight.data.clone()
        tok.type_id_embed.weight.data.zero_()
        out_no_typeid = tok(batch)["tokens"]
        tok.type_id_embed.weight.data.copy_(saved)
    diff = (out_normal - out_no_typeid).abs().max().item()
    print(f"  max |delta| zeroing type_id_embed: {diff:.4f}")
    assert diff > 1e-3, f"type_id has no effect (diff={diff})"
    print("  OK")


def test_pokemon_slot_disambiguation():
    print("\n== test 5: pokemon_slot_embed disambiguates slots ==")
    tok = _make_tokenizer().eval()
    collated = _load_sample_batch(1)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    saved = tok.species_embed.weight.data.clone()
    tok.species_embed.weight.data.zero_()
    with torch.no_grad():
        out2 = tok(batch)["tokens"]
    tok.species_embed.weight.data.copy_(saved)
    # Compare species token of our slot 0 vs slot 1 (indices 6 + 0*17 = 6 and 6 + 17 = 23).
    slot_diff = (out2[0, 6] - out2[0, 23]).abs().max().item()
    print(f"  with species_embed zeroed, slot-only diff = {slot_diff:.4f}")
    assert slot_diff > 1e-3, "pokemon_slot_embed doesn't disambiguate"
    print("  OK")


# ---------------- Determinism / correctness ----------------

def test_type_ordering_is_canonical():
    print("\n== test 6: type token ordering is canonical (sorted) ==")
    tok = _make_tokenizer(seed=123).eval()
    # Hand-craft a batch with one Pokemon that has two types in the multi-hot.
    fmt = tok.cfg.format_config
    B, T = 1, fmt.team_size
    cont = torch.zeros(B, T, _SL_TYPES[1] - _SL_TYPES[0]
                       + 0, dtype=torch.float32)  # placeholder shape; rebuilt below
    # Build full 285-dim cont, set just the type multi-hot at slot 0.
    from features import POKEMON_CONT_DIM
    cont = torch.zeros(B, T, POKEMON_CONT_DIM, dtype=torch.float32)
    # Set Pokemon 0 to types {2, 5}.
    cont[0, 0, 2] = 1.0
    cont[0, 0, 5] = 1.0
    # Pokemon 1 same types but reversed multi-hot order in source — which doesn't
    # actually affect the multi-hot (it's order-agnostic), but topk's tie-break
    # could pick (5, 2) for one and (2, 5) for another. Use the same set on both
    # and assert the type tokens are identical (with shared species etc.).
    cont[0, 1, 2] = 1.0
    cont[0, 1, 5] = 1.0

    ids = torch.zeros(B, T, 7, dtype=torch.long)
    ids[0, 0, 0] = 100
    ids[0, 1, 0] = 100   # same species → same type token if ordering is canonical
    banks = torch.zeros(B, T, 10, dtype=torch.long)

    with torch.no_grad():
        block = tok._encode_pokemon_block(ids, banks, cont)
    # type_token is the 4th of the 17 attribute tokens (index 3 after species/item/ability).
    type_idx_in_block = 3
    t0 = block[0, 0, type_idx_in_block]
    t1 = block[0, 1, type_idx_in_block]
    diff = (t0 - t1).abs().max().item()
    print(f"  same-types Pokemon 0 vs Pokemon 1 type-token diff = {diff:.2e}")
    assert diff < 1e-6, "type token differs across slots with identical types — non-canonical ordering!"
    print("  OK (sort canonicalizes order)")


def test_init_stats():
    print("\n== test 7: init weight stats ~ cfg.init_std=0.02 ==")
    tok = _make_tokenizer(seed=0)
    target = tok.cfg.init_std
    n_linear = n_embed = 0
    for m in tok.modules():
        if isinstance(m, torch.nn.Linear):
            std = m.weight.data.std().item()
            assert 0.4 * target < std < 1.6 * target, \
                f"Linear weight std {std:.4f} far from target {target}"
            n_linear += 1
        elif isinstance(m, torch.nn.Embedding):
            std = m.weight.data.std().item()
            assert 0.4 * target < std < 1.6 * target, \
                f"Embedding weight std {std:.4f} far from target {target}"
            n_embed += 1
    print(f"  checked {n_linear} Linear + {n_embed} Embedding weights -- std within [0.4x, 1.6x] of target")
    print("  OK")


def test_grad_flow_opp_threat():
    print("\n== test 8: gradient flows through opp_threat_unknown ==")
    tok = _make_tokenizer(seed=0).train()
    # Make sure the param requires grad and starts at zero.
    p = tok.opp_threat_unknown
    assert p.requires_grad
    assert (p == 0).all()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    out = tok(batch)
    loss = out["tokens"].sum()
    loss.backward()
    assert p.grad is not None, "opp_threat_unknown got no gradient"
    g = p.grad.abs().sum().item()
    print(f"  |grad| sum on opp_threat_unknown = {g:.3f}")
    assert g > 0, "gradient is exactly zero — param disconnected from forward graph"
    print("  OK")


def test_active_real_banks_override():
    print("\n== test 9: active_move_banks override changes slot-0 token (Postscript B) ==")
    tok = _make_tokenizer(seed=0).eval()
    collated = _load_sample_batch(2)

    # Without active_move_banks
    batch_lookup = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    batch_lookup_no_active = {k: v for k, v in batch_lookup.items() if k != "active_move_banks"}
    with torch.no_grad():
        out_lookup = tok(batch_lookup_no_active)["tokens"]

    # With active_move_banks (the trainer-supplied real banks)
    with torch.no_grad():
        out_real = tok(batch_lookup)["tokens"]

    # Slot-0 active 4 moves are at our-slot-0 attribute positions
    # 6 battle-state + 0*17 + 13..17 = 19..22.
    move_token_indices = list(range(6 + _FIRST_MOVE_OFFSET, 6 + _FIRST_MOVE_OFFSET + 4))
    diff_active_slot = (out_lookup[:, move_token_indices] - out_real[:, move_token_indices]).abs().max().item()
    # Other tokens should be unaffected (we only override slot-0 of our side).
    other_idx = [i for i in range(N_TOKENS) if i not in move_token_indices]
    diff_other = (out_lookup[:, other_idx] - out_real[:, other_idx]).abs().max().item()
    print(f"  diff at our-active move tokens : {diff_active_slot:.4f}")
    print(f"  diff at all other tokens       : {diff_other:.2e}")
    assert diff_active_slot > 1e-4, "real-banks override didn't change active-move tokens"
    assert diff_other < 1e-4, "override leaked to non-active-move tokens"
    print("  OK (slot-0 active moves change; rest unchanged)")


def test_doubles_rejected():
    print("\n== test 10: doubles format raises NotImplementedError ==")
    fmt_doubles = FormatConfig(
        battle_format="gen9doubles",
        n_active=2, n_bench=4, n_switches=4,
        n_moves=4, n_actions=8,
    )
    cfg = TransformerConfig.with_vocab_sizes_from_disk(format_config=fmt_doubles)
    try:
        Tokenizer(cfg)
    except NotImplementedError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("doubles config should have raised NotImplementedError")


if __name__ == "__main__":
    test_shape_and_finiteness()
    test_multiple_turns()
    test_unknown_opp_moves()
    test_type_id_embedding_contributes()
    test_pokemon_slot_disambiguation()
    test_type_ordering_is_canonical()
    test_init_stats()
    test_grad_flow_opp_threat()
    test_active_real_banks_override()
    test_doubles_rejected()
    print("\n=== all 10 tests passed ===")
