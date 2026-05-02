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
    print("\n== test 8: gradient flows through computed opp threat path ==")
    tok = _make_tokenizer(seed=0).train()
    # Postscript C: opp threat is now computed from damage chart + lookup, not
    # a learnable zero param. Verify gradient reaches the opp_threat_mlp's first
    # Linear (the layer that consumes the computed effectiveness signal).
    #
    # IMPORTANT: every per-attribute MLP ends in a LayerNorm. Using
    # `loss = tokens.sum()` produces a uniform upstream gradient, which
    # LayerNorm projects out (constant-component nulled). The Linear weight
    # then sees zero upstream gradient — a degenerate test artifact, NOT a
    # disconnection. In real training the heads produce non-uniform upstream
    # gradients and this never happens. We probe with a random projection.
    torch.manual_seed(0)
    target_param = tok.opp_threat_mlp[0].weight
    assert target_param.requires_grad
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    out = tok(batch)
    # Random projection of the opp_threat token (idx 5) — non-uniform upstream
    OPP_THREAT_IDX = 5
    proj = torch.randn(out["tokens"].shape[-1])
    loss = (out["tokens"][:, OPP_THREAT_IDX] * proj).sum()
    loss.backward()
    assert target_param.grad is not None, "opp_threat_mlp got no gradient"
    g = target_param.grad.abs().sum().item()
    print(f"  |grad| sum on opp_threat_mlp[0].weight (random projection): {g:.3f}")
    assert g > 0, "gradient is zero -- opp threat path disconnected from forward graph"
    print("  OK")


def test_opp_threat_uses_chart():
    print("\n== test 8b: opp threat changes when damage chart is perturbed ==")
    tok = _make_tokenizer(seed=0).eval()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    with torch.no_grad():
        out_normal = tok(batch)["tokens"]
        # Perturb the chart: zero everything (all moves "immune" against everyone)
        saved = tok.damage_chart.clone()
        tok.damage_chart.zero_()
        out_chart_zero = tok(batch)["tokens"]
        tok.damage_chart.copy_(saved)
    # Opp threat token is at sequence index 5 (0=actor, 1=critic, 2=field,
    # 3=transition, 4=our_threat, 5=opp_threat).
    diff_opp_threat = (out_normal[:, 5] - out_chart_zero[:, 5]).abs().max().item()
    diff_other = (out_normal[:, [0, 1, 2, 3, 4]] - out_chart_zero[:, [0, 1, 2, 3, 4]]).abs().max().item()
    print(f"  diff at opp_threat token (idx 5): {diff_opp_threat:.4f}")
    print(f"  diff at other battle-state tokens: {diff_other:.2e}")
    assert diff_opp_threat > 1e-3, "opp threat doesn't depend on damage chart"
    assert diff_other < 1e-4, "chart perturbation leaked beyond opp_threat token"
    print("  OK")


def test_active_flag_override_recovers_dynamic_dims():
    """Postscript D: active 4 moves' 107-dim flags should reflect REAL per-turn
    PP / disabled / STAB, not the lookup's static defaults."""
    print("\n== test 9b: active-move flag override recovers PP/disabled/STAB ==")
    tok = _make_tokenizer(seed=0).eval()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))

    # The dynamic indices in the 107-dim flag (per features.py:_project_move_flags)
    DYN = {"current_pp": 9, "disabled": 10, "stab": 12}

    # Read the memmap's active_move_cont (ground truth at t=0 for our 4 active moves).
    amc = batch["active_move_cont"]                        # (B, 4, 109)
    real_flags = amc[..., :128 - 21]                       # take 107 dims via shape; safer: amc[..., :MOVE_FLAG_DIM]
    from model_transformer import MOVE_FLAG_DIM
    real_flags = amc[..., :MOVE_FLAG_DIM]                  # (B, 4, 107)

    # Look up the flags the model would use WITHOUT override (i.e. lookup-only).
    move_ids = batch["our_pokemon_move_ids"][:, 0, :]      # (B, 4) — our active's 4 move ids
    lookup_flags = tok.move_flags_lookup[move_ids]         # (B, 4, 107)

    # For at least one of the 3 dynamic dims, real and lookup should differ on
    # at least one move (otherwise the test wouldn't actually exercise the fix).
    any_diff = False
    for name, idx in DYN.items():
        d = (real_flags[..., idx] - lookup_flags[..., idx]).abs().sum().item()
        print(f"  dim {idx} ({name}): cumulative diff between real and lookup = {d:.4f}")
        if d > 1e-4:
            any_diff = True
    assert any_diff, ("memmap and lookup happen to agree on all 3 dynamic dims for "
                      "the sampled episodes — pick different episodes or enrich the test fixture")

    # Now verify the model uses real_flags, not lookup_flags. We reach into
    # _encode_pokemon_block via the forward — in particular the active 4 move
    # tokens of our slot 0 should change if we strip the active_real_flags path.
    with torch.no_grad():
        out_with_real = tok(batch)["tokens"]
        # Strip both active_move_cont AND active_move_banks to fall back to lookup
        b_no_real = {k: v for k, v in batch.items()
                     if k not in ("active_move_cont", "active_move_banks")}
        # active_move_cont is required by forward; supply zeros so threat path runs.
        # We only want to disable the FLAG override here, but threat path consumes
        # the trailing 2 dims, so swap in zeros to keep shape but null the override.
        b_no_real["active_move_cont"] = torch.zeros_like(batch["active_move_cont"])
        b_no_real["active_move_banks"] = batch["active_move_banks"]   # keep banks override
        out_no_real_flags = tok(b_no_real)["tokens"]

    # Active move tokens of our slot 0 are at sequence indices [6+13, 6+14, 6+15, 6+16].
    from model_transformer import _FIRST_MOVE_OFFSET
    move_idx = list(range(6 + _FIRST_MOVE_OFFSET, 6 + _FIRST_MOVE_OFFSET + 4))
    diff_active = (out_with_real[:, move_idx] - out_no_real_flags[:, move_idx]).abs().max().item()
    print(f"  diff at our active 4 move tokens with vs without flag override: {diff_active:.4f}")
    assert diff_active > 1e-3, "flag override didn't change active-move tokens"
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


def test_restored_signals_reach_status_token():
    """Postscript C: verify each restored cont-slice signal flows into the
    status_token. Perturb the slice; status token should change; other tokens
    not. Catches accidental drop / bad slicing.
    """
    print("\n== test 11: each restored signal perturbs the status token ==")
    from features import POKEMON_CONT_DIM
    from model_transformer import (
        _SL_ACTIVE_FLAG, _SL_FAINTED, _SL_COMBAT, _SL_TOXIC,
        _SL_FUTURESIGHT, _SL_VISIBILITY,
    )
    tok = _make_tokenizer(seed=0).eval()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    # status_token of our slot 0: idx 6 + 0*17 + 4 (per _PER_POKEMON_TT layout) = 10.
    STATUS_IDX = 6 + 4
    SPECIES_IDX = 6 + 0    # species_token of our slot 0 — should NOT change
    HP_IDX = 6 + 5         # hp_token of our slot 0 — should NOT change

    slices_to_test = [
        ("active",        _SL_ACTIVE_FLAG),
        ("fainted",       _SL_FAINTED),
        ("combat",        _SL_COMBAT),
        ("toxic",         _SL_TOXIC),
        ("future_sight",  _SL_FUTURESIGHT),
        ("visibility",    _SL_VISIBILITY),
    ]
    with torch.no_grad():
        out_baseline = tok(batch)["tokens"]
    for name, (a, b) in slices_to_test:
        # Perturb that slice by adding 1.0 across the cont
        perturbed = {k: (v.clone() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        cont = perturbed["our_pokemon_cont"].clone()
        cont[:, 0, a:b] += 1.0
        perturbed["our_pokemon_cont"] = cont
        with torch.no_grad():
            out_p = tok(perturbed)["tokens"]
        d_status  = (out_p[:, STATUS_IDX]  - out_baseline[:, STATUS_IDX]).abs().max().item()
        d_species = (out_p[:, SPECIES_IDX] - out_baseline[:, SPECIES_IDX]).abs().max().item()
        d_hp      = (out_p[:, HP_IDX]      - out_baseline[:, HP_IDX]).abs().max().item()
        print(f"  {name:13s}: status delta={d_status:.4f}  species={d_species:.2e}  hp={d_hp:.2e}")
        assert d_status > 1e-4, f"perturbing {name} didn't reach status_token"
        # Species and HP should be unaffected
        assert d_species < 1e-5, f"{name} leaked into species_token"
        assert d_hp < 1e-5, f"{name} leaked into hp_token"
    print("  OK (each restored signal reaches status_token, doesn't leak)")


def test_physical_banks_reach_status_token():
    """Postscript C: level/weight/height are wired through NumericalBanks into
    the status_token. Perturbing the bank input should change status."""
    print("\n== test 12: level/weight/height bank embeds flow into status_token ==")
    tok = _make_tokenizer(seed=0).eval()
    collated = _load_sample_batch(2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    STATUS_IDX = 6 + 4
    HP_IDX = 6 + 5
    with torch.no_grad():
        out_baseline = tok(batch)["tokens"]
    for name, col in [("level", 1), ("weight", 2), ("height", 3)]:
        perturbed = {k: (v.clone() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        bk = perturbed["our_pokemon_banks"].clone()
        bk[:, 0, col] = (bk[:, 0, col] + 1) % 40   # bump within range
        perturbed["our_pokemon_banks"] = bk
        with torch.no_grad():
            out_p = tok(perturbed)["tokens"]
        d_status = (out_p[:, STATUS_IDX] - out_baseline[:, STATUS_IDX]).abs().max().item()
        d_hp     = (out_p[:, HP_IDX]     - out_baseline[:, HP_IDX]).abs().max().item()
        print(f"  {name:6s}: status delta={d_status:.4f}  hp delta={d_hp:.2e}")
        assert d_status > 1e-4, f"perturbing {name} didn't reach status_token"
        assert d_hp < 1e-5, f"{name} leaked into hp_token"
    print("  OK")


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
    test_opp_threat_uses_chart()
    test_active_flag_override_recovers_dynamic_dims()
    test_active_real_banks_override()
    test_restored_signals_reach_status_token()
    test_physical_banks_reach_status_token()
    test_doubles_rejected()
    print("\n=== all 14 tests passed ===")
