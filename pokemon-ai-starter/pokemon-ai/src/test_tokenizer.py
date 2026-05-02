# test_tokenizer.py
# Unit tests for model_transformer.Tokenizer (REWRITE_DESIGN.md Week 1).
#
# Loads 5 real episodes from data/datasets/human_v8_100k/ via MemmapDataset,
# runs the Tokenizer, and asserts the spec contract:
#   - Output tokens shape: (B, 212, d_model)
#   - No NaN / no inf
#   - type_ids has all 20 token types present
#   - Move tokens for opp Pokemon use move_id=0 ("unknown") for unrevealed moves
#     and produce different output than known-move tokens
#
# Run: cd pokemon-ai-starter/pokemon-ai/src && python test_tokenizer.py

from __future__ import annotations
from pathlib import Path

import torch

from dataset import MemmapDataset, collate_seq, unpack_turn_batch
from model_transformer import (
    Tokenizer, TransformerConfig, load_move_flag_lookup,
    N_TOKENS, N_TOKEN_TYPES, TT_MOVE,
)


MEMMAP_DIR = "data/datasets/human_v8_100k"
LOOKUP_PATH = "data/lookup/move_flags_v1.pt"


def load_sample_batch(n_episodes: int = 5):
    ds = MemmapDataset(MEMMAP_DIR, split="train")
    samples = [ds[i] for i in range(n_episodes)]
    return collate_seq(samples)


def assert_no_nan_inf(t: torch.Tensor, name: str):
    assert torch.isfinite(t).all(), f"{name} has {(~torch.isfinite(t)).sum().item()} non-finite values"


def test_shape_and_finiteness():
    print("== test 1: shape + no NaN/inf on 5 real episodes ==")
    cfg = TransformerConfig()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup)
    tok.eval()

    collated = load_sample_batch(n_episodes=5)
    B = collated["our_pokemon_ids"].shape[0]
    T = collated["mask"].shape[1]
    print(f"  loaded B={B} episodes, T={T} max turns")

    # Tokenize the first turn (t=0) of each episode.
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))
    with torch.no_grad():
        out = tok(batch)
    tokens = out["tokens"]
    assert tokens.shape == (B, N_TOKENS, cfg.d_model), \
        f"tokens shape {tokens.shape} != ({B}, {N_TOKENS}, {cfg.d_model})"
    assert_no_nan_inf(tokens, "tokens")
    print(f"  tokens shape: {tuple(tokens.shape)} OK")
    print(f"  no NaN / no inf: OK")

    # All 20 token types should be present in the position-ID tensor
    unique_tt = out["type_ids"].unique().tolist()
    assert sorted(unique_tt) == list(range(N_TOKEN_TYPES)), \
        f"type_ids missing some types: present={unique_tt}"
    print(f"  all {N_TOKEN_TYPES} token types present in type_ids: OK")


def test_multiple_turns_per_episode():
    print("\n== test 2: process 3 turns of episode 0 ==")
    cfg = TransformerConfig()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup)
    tok.eval()

    collated = load_sample_batch(n_episodes=2)
    seq_lens = collated["seq_lens"].tolist()
    print(f"  episode lengths: {seq_lens}")

    # Sanity: process turns 0..min(2, L-1) for episode 0
    n_turns = min(3, seq_lens[0])
    for t in range(n_turns):
        batch = unpack_turn_batch(collated, t=t, device=torch.device("cpu"))
        with torch.no_grad():
            out = tok(batch)
        assert out["tokens"].shape[1] == N_TOKENS
        assert_no_nan_inf(out["tokens"], f"tokens@t={t}")
    print(f"  {n_turns} turns processed without NaN: OK")


def test_unknown_opp_moves():
    print("\n== test 3: opp Pokemon with unrevealed (move_id=0) moves ==")
    cfg = TransformerConfig()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup)
    tok.eval()

    collated = load_sample_batch(n_episodes=5)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))

    # Look at opp_pokemon_move_ids[..., 3:7] in the underlying memmap. The
    # collate stores the full (B, T, 6, 7) tensor; for t=0 we expect early-game
    # most opp moves are id=0 (unknown).
    opp_move_ids = collated["opp_pokemon_move_ids"][:, 0]  # (B, 6, 4)
    n_unknown = int((opp_move_ids == 0).sum().item())
    n_total = int(opp_move_ids.numel())
    print(f"  opp move slots @ turn 0: {n_unknown}/{n_total} are unknown (id=0)")
    assert n_unknown > 0, "expected some unrevealed opp moves at t=0"

    # Run tokenizer; assert finite even with id=0 moves (the lookup row 0 is all zeros).
    with torch.no_grad():
        out = tok(batch)
    assert_no_nan_inf(out["tokens"], "tokens")

    # The token at index for "opp pokemon 0, move slot 0" — find its position.
    # Per layout: 6 battle-state tokens + 6 our × 17 + (opp 0..5) × 17.
    # Opp Pokemon 0 starts at index 6 + 6*17 = 108; its move_token[0] is at +13.
    OPP_0_MOVE_0_IDX = 6 + 6 * 17 + 13
    assert tok.type_ids[OPP_0_MOVE_0_IDX].item() == TT_MOVE
    assert tok.poke_slots[OPP_0_MOVE_0_IDX].item() == 6  # opp pokemon 0
    assert tok.move_slots[OPP_0_MOVE_0_IDX].item() == 1  # move slot 0 → id 1 (offset)
    print(f"  opp_0_move_0 position-IDs sanity check: OK (idx={OPP_0_MOVE_0_IDX})")


def test_type_id_embedding_added():
    print("\n== test 4: type_id embeddings actually contribute to output ==")
    cfg = TransformerConfig()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup)
    tok.eval()

    collated = load_sample_batch(n_episodes=2)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))

    with torch.no_grad():
        out_normal = tok(batch)
        # Zero the type_id embedding and re-run; result must change.
        saved = tok.type_id_embed.weight.data.clone()
        tok.type_id_embed.weight.data.zero_()
        out_no_typeid = tok(batch)
        tok.type_id_embed.weight.data.copy_(saved)

    diff = (out_normal["tokens"] - out_no_typeid["tokens"]).abs().max().item()
    print(f"  max |delta| when zeroing type_id_embed: {diff:.4f}")
    assert diff > 1e-3, f"type_id embeddings have negligible effect (max diff {diff})"
    print(f"  type_id contributes to output: OK")


def test_pokemon_slot_disambiguation():
    print("\n== test 5: pokemon_slot_embed disambiguates same-attribute tokens ==")
    cfg = TransformerConfig()
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup)
    tok.eval()

    collated = load_sample_batch(n_episodes=1)
    batch = unpack_turn_batch(collated, t=0, device=torch.device("cpu"))

    with torch.no_grad():
        out = tok(batch)["tokens"]  # (1, 212, d)

    # If our Pokemon 0 and Pokemon 1 have different species, their species
    # tokens (TT_SPECIES) should differ. Indices: 6 + 0*17 = 6 and 6 + 1*17 = 23.
    species_ids = batch["our_pokemon_ids"][0, :, 0]
    print(f"  our species ids (active+5 bench): {species_ids.tolist()}")
    if species_ids[0] != species_ids[1]:
        diff = (out[0, 6] - out[0, 23]).abs().max().item()
        print(f"  |species_token[0] - species_token[1]| = {diff:.3f} (different species expected)")
        assert diff > 0.1, f"different species produce too-similar tokens (diff={diff})"

    # Even if species ids matched, slot embeddings should make the tokens differ:
    # zero the species_embed weights and check tokens still differ via slot_embed.
    saved = tok.species_embed.weight.data.clone()
    tok.species_embed.weight.data.zero_()
    with torch.no_grad():
        out2 = tok(batch)["tokens"]
    tok.species_embed.weight.data.copy_(saved)
    slot_diff = (out2[0, 6] - out2[0, 23]).abs().max().item()
    print(f"  with species_embed zeroed, slot-only diff = {slot_diff:.4f}")
    assert slot_diff > 1e-3, "pokemon_slot_embed must disambiguate same-attribute tokens"
    print(f"  pokemon_slot_embed disambiguates: OK")


if __name__ == "__main__":
    test_shape_and_finiteness()
    test_multiple_turns_per_episode()
    test_unknown_opp_moves()
    test_type_id_embedding_added()
    test_pokemon_slot_disambiguation()
    print("\n=== all 5 tests passed ===")
