# bench_tokenizer.py
# Forward-pass benchmark per REWRITE_DESIGN.md Week 1 sub-task #6:
# "On RTX 3060 Laptop, time Tokenizer + dummy_spatial(212, d_model=256) on a
#  batch of 32 turns. Run 10 iterations after warmup, report median.
#  Target: <50ms median. If >100ms, profile and optimize."
#
# Run: cd pokemon-ai-starter/pokemon-ai/src && python bench_tokenizer.py

from __future__ import annotations
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import MemmapDataset, collate_seq, unpack_turn_batch
from model_transformer import (
    Tokenizer, TransformerConfig, load_move_flag_lookup,
    N_TOKENS,
)


MEMMAP_DIR = "data/datasets/human_v8_100k"
LOOKUP_PATH = "data/lookup/move_flags_v1.pt"
BATCH_TURNS = 32
WARMUP = 3
ITERS = 10


def make_dummy_spatial(d_model: int, n_layers: int = 6, n_heads: int = 8,
                      ff_mult: int = 4, dropout: float = 0.05) -> nn.Module:
    """Stand-in for the Week 2 SpatialTransformer; same dims so timings are realistic."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads,
        dim_feedforward=d_model * ff_mult, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


def gather_batch_of_n_turns(collated: dict, n: int, device: torch.device) -> dict:
    """Pull n distinct (b, t) turn-batches from collated and stack them as one
    spatial batch. We use turn 0 of n different episodes."""
    # collated has B episodes, each with T turns. We just take the first n of t=0.
    B = collated["our_pokemon_ids"].shape[0]
    assert B >= n, f"need at least {n} episodes, got {B}"
    batch = unpack_turn_batch(collated, t=0, device=device)
    # batch is already (B, ...). Slice to first n.
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[:n] for kk, vv in v.items()}
        else:
            out[k] = v[:n]
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")

    cfg = TransformerConfig()
    print(f"Config: d_model={cfg.d_model}, n_spatial_layers={cfg.n_spatial_layers}, "
          f"n_heads={cfg.n_heads}, n_summary_tokens={cfg.n_summary_tokens}")

    print("Loading lookup + tokenizer...")
    lookup = load_move_flag_lookup(Path(LOOKUP_PATH))
    tok = Tokenizer(cfg, move_flag_lookup=lookup).to(device).eval()
    spatial = make_dummy_spatial(
        cfg.d_model, cfg.n_spatial_layers, cfg.n_heads, cfg.ff_mult, cfg.dropout,
    ).to(device).eval()
    print(f"  Tokenizer params:      {sum(p.numel() for p in tok.parameters()):>10,}")
    print(f"  Dummy spatial params:  {sum(p.numel() for p in spatial.parameters()):>10,}")

    print(f"Loading {BATCH_TURNS} sample turns from {MEMMAP_DIR}...")
    ds = MemmapDataset(MEMMAP_DIR, split="train")
    eps = [ds[i] for i in range(BATCH_TURNS)]
    collated = collate_seq(eps)
    batch = gather_batch_of_n_turns(collated, n=BATCH_TURNS, device=device)
    print(f"  batch ready: {BATCH_TURNS} turns")

    # ---- Warmup ----
    for _ in range(WARMUP):
        with torch.no_grad():
            tokens = tok(batch)["tokens"]
            _ = spatial(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- Timed ----
    timings_tok = []
    timings_total = []
    for it in range(ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            tokens = tok(batch)["tokens"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = spatial(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings_tok.append((t1 - t0) * 1000.0)
        timings_total.append((t2 - t0) * 1000.0)
        print(f"  iter {it}:  tokenizer {timings_tok[-1]:6.2f} ms  +  spatial {(t2-t1)*1000:6.2f} ms  "
              f"=  total {timings_total[-1]:6.2f} ms")

    median_tok = statistics.median(timings_tok)
    median_total = statistics.median(timings_total)
    print(f"\n=== Median over {ITERS} iters (B={BATCH_TURNS} turns) ===")
    print(f"  Tokenizer alone:        {median_tok:6.2f} ms")
    print(f"  Tokenizer + spatial:    {median_total:6.2f} ms")
    print(f"  Per-turn cost:          {median_total / BATCH_TURNS:6.3f} ms")
    print(f"\nTarget: < 50 ms median (warning at >100 ms).")
    if median_total < 50:
        print("  PASS: comfortably under target.")
    elif median_total < 100:
        print("  CLOSE: between 50-100 ms — acceptable but consider Week 1 optimization.")
    else:
        print("  WARN: > 100 ms — profile before Week 2.")


if __name__ == "__main__":
    main()
