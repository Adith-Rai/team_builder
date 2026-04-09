"""Test that batched temporal inference matches per-item temporal inference.

Loads a real checkpoint, creates synthetic histories of varying lengths,
and compares the batched path vs per-item path for numerical equivalence.
"""
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model import PokeTransformer, PokeTransformerConfig


def test_temporal_batching():
    """Test batched temporal matches per-item temporal."""
    # Find a checkpoint to load
    ckpt_path = None
    for candidate in [
        "data/models/rl_v9/selfplay_v9_20260404_192922/snapshot_1164.pt",
        "data/models/rl_v9/selfplay_v9_20260404_192922/snapshot_1159.pt",
    ]:
        if os.path.exists(candidate):
            ckpt_path = candidate
            break

    if ckpt_path is None:
        print("ERROR: No checkpoint found for testing")
        return False

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PokeTransformerConfig(**ckpt["model_config"])
    model = PokeTransformer(cfg)

    # Handle dim expansion
    sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    for k in model_sd:
        if k in sd and sd[k].shape != model_sd[k].shape:
            print(f"  Skipping mismatched key: {k} ({sd[k].shape} vs {model_sd[k].shape})")
            sd[k] = model_sd[k]
    model.load_state_dict(sd, strict=False)
    model.eval()

    D = cfg.d_model
    max_ctx = cfg.temporal_context

    # Test cases: different history lengths
    test_cases = [
        ("empty history", 0),
        ("1 turn", 1),
        ("5 turns", 5),
        ("15 turns (typical)", 15),
        ("50 turns (long game)", 50),
        ("199 turns (near max)", 199),
    ]

    print(f"\nModel d_model={D}, temporal_context={max_ctx}")
    print(f"Running {len(test_cases)} test cases...\n")

    all_passed = True

    for name, h_len in test_cases:
        # Create random history and current summary
        if h_len > 0:
            history = torch.randn(1, h_len, D)
        else:
            history = None
        current_summary = torch.randn(D)

        # === Per-item path (old code) ===
        with torch.no_grad():
            summary_i = current_summary.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            if history is not None and history.shape[1] > 0:
                all_summ = torch.cat([history, summary_i], dim=1)
            else:
                all_summ = summary_i
            per_item_ctx = model.temporal(all_summ)  # (1, D)

        # === Batched path (new code, simulating N=1) ===
        with torch.no_grad():
            seq_len = h_len + 1
            max_T = min(seq_len, max_ctx)
            seq_lens_t = torch.tensor([seq_len], dtype=torch.long).clamp(max=max_T)

            padded = torch.zeros(1, max_T, D)
            if history is not None and history.shape[1] > 0:
                h = history.squeeze(0)  # (h_len, D)
                if h.shape[0] + 1 > max_T:
                    h = h[-(max_T - 1):]
                h_actual = h.shape[0]
                padded[0, :h_actual] = h
                padded[0, h_actual] = current_summary
            else:
                padded[0, 0] = current_summary

            batched_ctx = model.temporal(padded.float(), seq_lens_t)  # (1, D)

        diff = (per_item_ctx - batched_ctx).abs().max().item()
        passed = diff < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name:30s} h_len={h_len:4d}  max_diff={diff:.2e}")

    # Test with multiple items in a batch (different lengths)
    print(f"\n  Testing multi-item batch (N=5, mixed lengths)...")
    lengths = [0, 3, 15, 40, 100]
    histories = []
    summaries_list = []
    per_item_results = []

    for h_len in lengths:
        h = torch.randn(1, h_len, D) if h_len > 0 else None
        s = torch.randn(D)
        histories.append(h)
        summaries_list.append(s)

        # Per-item reference
        with torch.no_grad():
            si = s.unsqueeze(0).unsqueeze(1)
            if h is not None and h.shape[1] > 0:
                all_s = torch.cat([h, si], dim=1)
            else:
                all_s = si
            ctx = model.temporal(all_s)
            per_item_results.append(ctx.squeeze(0))

    # Batched
    N = len(lengths)
    seq_lens = [h_len + 1 for h_len in lengths]
    max_T = min(max(seq_lens), max_ctx)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long).clamp(max=max_T)
    padded = torch.zeros(N, max_T, D)

    for i, (h, s, h_len) in enumerate(zip(histories, summaries_list, lengths)):
        if h is not None and h.shape[1] > 0:
            hh = h.squeeze(0)
            if hh.shape[0] + 1 > max_T:
                hh = hh[-(max_T - 1):]
            actual = hh.shape[0]
            padded[i, :actual] = hh
            padded[i, actual] = s
        else:
            padded[i, 0] = s

    with torch.no_grad():
        batched_results = model.temporal(padded.float(), seq_lens_t)  # (N, D)

    for i, (h_len, per_item_r) in enumerate(zip(lengths, per_item_results)):
        diff = (per_item_r - batched_results[i]).abs().max().item()
        passed = diff < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] batch item {i} (h_len={h_len:4d})  max_diff={diff:.2e}")

    print(f"\n{'='*50}")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — do not use batched temporal!")
    print(f"{'='*50}")
    return all_passed


if __name__ == "__main__":
    test_temporal_batching()
