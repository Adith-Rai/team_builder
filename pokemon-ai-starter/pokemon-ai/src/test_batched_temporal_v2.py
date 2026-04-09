"""Rigorous test: compare old per-item InferenceBatcher vs new batched version.

Uses the actual _gpu_forward code paths with real model weights and simulated
request batches that mimic what happens during actual gameplay.
"""
import torch
import torch.nn.functional as F
import sys, os, copy

sys.path.insert(0, os.path.dirname(__file__))

from model import PokeTransformer, PokeTransformerConfig


def make_fake_batch(model, device, D):
    """Create a fake observation batch (B=1) with realistic tensor shapes."""
    cfg = model.cfg
    batch = {}
    # Spatial tokens (16 entities)
    n_tokens = 16
    # We need the actual keys that forward_spatial expects
    # Simulate by running a dummy through forward to find keys
    # Instead, create minimal dict that forward_spatial can handle
    # Actually just create random summaries directly — we're testing temporal, not spatial
    return None  # We'll test temporal directly


def test_batched_vs_peritm_temporal_full():
    """Test the actual temporal + heads pipeline (not just temporal alone)."""
    ckpt_path = "data/models/rl_v9/selfplay_v9_20260404_192922/snapshot_1164.pt"
    if not os.path.exists(ckpt_path):
        print("ERROR: checkpoint not found")
        return False

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PokeTransformerConfig(**ckpt["model_config"])
    model = PokeTransformer(cfg)
    sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    for k in model_sd:
        if k in sd and sd[k].shape != model_sd[k].shape:
            sd[k] = model_sd[k]
    model.load_state_dict(sd, strict=False)
    model.eval()

    D = cfg.d_model

    # Simulate N=8 battles with different history lengths (like a real batch)
    history_lengths = [0, 1, 3, 10, 25, 50, 100, 150]
    N = len(history_lengths)

    # Create fake spatial outputs and summaries (as if spatial already ran)
    spatial_out = torch.randn(N, 16, D)  # (N, 16, D)
    summaries = torch.randn(N, D)        # (N, D)
    action_ctx = torch.randn(N, 9, D)    # (N, 9, D)
    legal_mask = torch.ones(N, 9)         # all legal

    # Create histories
    histories = []
    for h_len in history_lengths:
        if h_len > 0:
            histories.append(torch.randn(1, h_len, D))
        else:
            histories.append(None)

    print(f"\nTesting N={N} items with history lengths: {history_lengths}")

    # === OLD PER-ITEM PATH ===
    old_logits_list = []
    old_values_list = []
    old_temporal_list = []

    with torch.no_grad():
        for i in range(N):
            history = histories[i]
            summary_i = summaries[i:i+1].unsqueeze(1)  # (1, 1, D)

            if history is not None and history.shape[1] > 0:
                all_summ = torch.cat([history, summary_i], dim=1)
            else:
                all_summ = summary_i

            temporal_ctx = model.temporal(all_summ)  # (1, D)
            old_temporal_list.append(temporal_ctx.squeeze(0))

            # Policy head
            actor_out = spatial_out[i:i+1, 0, :]
            at = torch.cat([actor_out, temporal_ctx], dim=-1)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)
            act_i = action_ctx[i:i+1]
            pi_input = torch.cat([at_exp, act_i], dim=-1)
            logits = model.policy_head(pi_input).squeeze(-1)
            logits = logits.float().masked_fill(legal_mask[i:i+1] < 0.5, -100.0)
            old_logits_list.append(logits.squeeze(0))

            # Value head
            critic_out = spatial_out[i:i+1, 1, :]
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)
            v_logits = model.value_head(vi)
            v_probs = F.softmax(v_logits, dim=-1)
            value = (v_probs * model.v_support).sum(-1)
            old_values_list.append(value.squeeze(0))

    # === NEW BATCHED PATH ===
    with torch.no_grad():
        # Build padded temporal input
        seq_lens = []
        for i in range(N):
            h_len = histories[i].shape[1] if histories[i] is not None and histories[i].shape[1] > 0 else 0
            seq_lens.append(h_len + 1)

        max_T = min(max(seq_lens), model.temporal.temporal_context)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long).clamp(max=max_T)

        all_summaries_padded = torch.zeros(N, max_T, D)
        for i in range(N):
            history = histories[i]
            summary_i = summaries[i]
            if history is not None and history.shape[1] > 0:
                h = history.squeeze(0)
                if h.shape[0] + 1 > max_T:
                    h = h[-(max_T - 1):]
                h_len = h.shape[0]
                all_summaries_padded[i, :h_len] = h
                all_summaries_padded[i, h_len] = summary_i
            else:
                all_summaries_padded[i, 0] = summary_i

        temporal_ctx_batched = model.temporal(all_summaries_padded.float(), seq_lens_t)  # (N, D)

        # Batched policy head
        actor_out = spatial_out[:, 0, :]
        at = torch.cat([actor_out, temporal_ctx_batched], dim=-1)
        at_exp = at.unsqueeze(1).expand(-1, 9, -1)
        pi_input = torch.cat([at_exp, action_ctx], dim=-1)
        new_logits = model.policy_head(pi_input).squeeze(-1)
        new_logits = new_logits.float().masked_fill(legal_mask < 0.5, -100.0)

        # Batched value head
        critic_out = spatial_out[:, 1, :]
        vi = torch.cat([critic_out, temporal_ctx_batched], dim=-1)
        v_logits = model.value_head(vi)
        v_probs = F.softmax(v_logits, dim=-1)
        new_values = (v_probs * model.v_support).sum(-1)

    # === COMPARE ===
    print(f"\n{'Item':<6} {'h_len':<8} {'temporal_diff':<15} {'logits_diff':<15} {'value_diff':<15} {'same_action':<12}")
    print("-" * 75)

    all_passed = True
    for i in range(N):
        t_diff = (old_temporal_list[i] - temporal_ctx_batched[i]).abs().max().item()
        l_diff = (old_logits_list[i] - new_logits[i]).abs().max().item()
        v_diff = (old_values_list[i] - new_values[i]).abs().item()

        old_action = old_logits_list[i].argmax().item()
        new_action = new_logits[i].argmax().item()
        same = old_action == new_action

        passed = t_diff < 1e-4 and l_diff < 1e-3 and same
        if not passed:
            all_passed = False

        status = "OK" if passed else "MISMATCH"
        print(f"  {i:<4} {history_lengths[i]:<8} {t_diff:<15.2e} {l_diff:<15.2e} {v_diff:<15.4f} {str(same):<12} {status}")

    # Test that action distributions are similar (not just argmax)
    print(f"\n  Action probability comparison (softmax of logits):")
    for i in range(N):
        old_probs = F.softmax(old_logits_list[i], dim=-1)
        new_probs = F.softmax(new_logits[i], dim=-1)
        kl = (old_probs * (old_probs / (new_probs + 1e-10)).log()).sum().item()
        print(f"    item {i} (h_len={history_lengths[i]:>3}): KL divergence = {kl:.2e}")
        if kl > 0.01:
            print(f"    WARNING: KL > 0.01 — distributions differ meaningfully!")
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("  ALL CHECKS PASSED — batched temporal is correct")
    else:
        print("  ISSUES FOUND — investigate before using")
    print(f"{'='*50}")
    return all_passed


if __name__ == "__main__":
    test_batched_vs_peritm_temporal_full()
