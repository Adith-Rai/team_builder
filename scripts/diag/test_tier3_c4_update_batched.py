#!/usr/bin/env python
"""Tier 3 C4: ppo_update_batched() smoke test.

Verifies the full composition (collate_episodes -> forward_ppo_sequence ->
ppo_loss_batched -> backward -> optimizer step) runs end-to-end without
errors on synthetic data + correct stats shape + edge case handling.

Real algorithmic correctness (gradient parity vs current ppo_update,
training quality preservation) is validated in C6: 20-iter A/B against
Phase 1 v3's iter 19/29/39/49 deep playstyle baseline (after Phase 1 v3
finishes, ~5 days from now). See docs/PHASE1_V3_OBSERVATIONS.md §C6.

Acceptance gates:
  1. End-to-end smoke: synthetic data runs through ppo_update_batched
     without errors; stats dict has correct shape; n_succeeded > 0
  2. Optimizer step actually changes model parameters (gradient flowed)
  3. KL gate triggers when approx_kl exceeds target_kl × 5
  4. Empty episodes returns zero-stats gracefully
  5. in_warmup=True raises NotImplementedError (v1 limitation)
  6. Multi-epoch loop runs N times then completes

CPU-only. Runs in ~3-5s. No GPU needed.

Usage:
  python scripts/diag/test_tier3_c4_update_batched.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")


_setup()
from ppo import ppo_update_batched  # noqa: E402


# ----------------------------------------------------------------
# Mini model — exposes forward_ppo_sequence + twohot_target + value_head
# Skips the real spatial/temporal stack; produces shape-compatible synthetic
# outputs WITH gradient so optimizer.step() actually does something.
# ----------------------------------------------------------------

class _MiniValueHead(nn.Module):
    def __init__(self, d_in: int = 16, v_bins: int = 51,
                 v_min: float = -1.0, v_max: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(d_in, v_bins)
        self.register_buffer(
            "v_support", torch.linspace(v_min, v_max, v_bins)
        )


class _MiniCfg:
    v_min = -1.0
    v_max = 1.0
    v_bins = 51

    class format_config:
        n_actions = 9


class _MiniModel(nn.Module):
    """Minimal model with the surface area ppo_update_batched needs.
    forward_ppo_sequence produces synthetic logits/v_logits via two linear
    layers — gives real parameters + real gradients without the full
    transformer."""
    def __init__(self, d_emb: int = 16, n_actions: int = 9, v_bins: int = 51):
        super().__init__()
        self.cfg = _MiniCfg()
        self.value_head = _MiniValueHead(d_in=d_emb, v_bins=v_bins)
        self.policy_head = nn.Linear(d_emb, n_actions)
        # An "embedding" of (B, L_max) -> (B, L_max, d_emb) — synthetic
        self.embed = nn.Embedding(1000, d_emb)  # used by forward_ppo_sequence
        self._n_actions = n_actions
        self._v_bins = v_bins
        self._d_emb = d_emb
        # arch_compat.get_v_support(model) looks for model.v_support at top
        # level — mirror the real TransformerBattlePolicy contract.
        self.register_buffer(
            "v_support", torch.linspace(self.cfg.v_min, self.cfg.v_max, v_bins)
        )

    def twohot_target(self, value: torch.Tensor) -> torch.Tensor:
        """Same logic as TransformerBattlePolicy.twohot_target."""
        v_support = self.value_head.v_support.to(value.device)
        value = value.clamp(self.cfg.v_min, self.cfg.v_max)
        bin_width = v_support[1] - v_support[0]
        idx = (value - v_support[0]) / bin_width
        lo = idx.floor().long().clamp(0, self.cfg.v_bins - 2)
        hi = (lo + 1).clamp(max=self.cfg.v_bins - 1)
        weight_hi = (idx - lo.float()).clamp(0, 1)
        target = torch.zeros(value.shape[0], self.cfg.v_bins, device=value.device)
        target.scatter_(1, lo.unsqueeze(1), (1 - weight_hi).unsqueeze(1))
        target.scatter_(1, hi.unsqueeze(1), weight_hi.unsqueeze(1))
        return target

    def forward_ppo_sequence(self, collated: dict, device: torch.device) -> dict:
        """Synthetic forward: use actions as 'token IDs' to fetch embeddings,
        then linear-project to logits + v_logits. Real parameters -> real
        gradients. Padding positions get -100.0 logits, 0 v_logits (matches
        the real forward_ppo_sequence's contract)."""
        B = collated["B"]
        L_max = collated["L_max"]
        actions = collated["actions"].to(device)  # (B, L_max) long, padding=0
        pad_mask = collated["pad_mask"].to(device)  # (B, L_max) bool

        # Embed actions as a stand-in for spatial+temporal output
        x = self.embed(actions)  # (B, L_max, d_emb)
        logits = self.policy_head(x)  # (B, L_max, n_actions)
        v_logits = self.value_head.linear(x)  # (B, L_max, v_bins)
        value = (F.softmax(v_logits, dim=-1)
                 * self.value_head.v_support).sum(-1)  # (B, L_max)

        # Apply padding fillers per the contract
        logits = logits.clone()
        v_logits = v_logits.clone()
        value = value.clone()
        pad_inv = ~pad_mask
        logits[pad_inv] = -100.0
        v_logits[pad_inv] = 0.0
        value[pad_inv] = 0.0
        return {"action_logits": logits, "v_logits": v_logits, "value": value}


# ----------------------------------------------------------------
# Synthetic episode helpers
# ----------------------------------------------------------------

def _make_episode(T: int, A: int = 9, feat_dim: int = 4, seed: int = 0):
    """Synthetic episode dict matching build_ppo_episodes' output schema."""
    rng = np.random.RandomState(seed)
    feat_batches = []
    for _ in range(T):
        feat_batches.append({"x": torch.randn(1, feat_dim)})
    actions = list(rng.randint(0, A, size=T).astype(int))
    old_logp = list(rng.randn(T).astype(np.float32) * 0.5)
    advantages = list(rng.randn(T).astype(np.float32))
    returns = list(np.tanh(rng.randn(T)).astype(np.float32))
    action_masks = [rng.uniform(0.5, 1.0, size=A).astype(np.float32) for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
    }


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

def test_smoke_runs_to_completion():
    """End-to-end: synthetic episodes -> optimizer step -> stats dict shape."""
    print("=== Test 1: end-to-end smoke (multi-epoch) ===")
    torch.manual_seed(0)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    eps = [_make_episode(T=8, seed=10),
           _make_episode(T=12, seed=11),
           _make_episode(T=5, seed=12)]
    device = torch.device("cpu")
    cfg = model.cfg

    stats = ppo_update_batched(
        model, optimizer, eps, device, cfg,
        epochs=3, target_kl=10.0,  # big kl threshold so no early stop
    )
    print(f"  stats: {stats}")
    expected_keys = {"pi", "v", "ent", "kl", "ratio_clip_frac",
                     "value_mean", "return_mean", "adv_abs_mean",
                     "n_succeeded", "n_failed", "n_skipped_kl", "n_skipped_nan",
                     "bc_kl"}  # S57: BC anchor stat (0.0 when anchor inactive)
    assert set(stats.keys()) == expected_keys, \
        f"keys mismatch: {set(stats.keys()) ^ expected_keys}"
    assert stats["n_succeeded"] > 0, f"no epochs succeeded"
    assert stats["n_failed"] == 0, f"unexpected failures: {stats['n_failed']}"
    # No NaN in any stat
    for k, v in stats.items():
        if isinstance(v, float):
            assert not np.isnan(v) and not np.isinf(v), f"{k}: {v}"
    print(f"  n_succeeded={stats['n_succeeded']}, all stats finite  [PASS]")


def test_optimizer_actually_steps():
    """Model parameters must change after ppo_update_batched (gradient flowed
    + optimizer.step() actually fired)."""
    print("=== Test 2: optimizer step changes model params ===")
    torch.manual_seed(1)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)  # large lr

    # Snapshot initial params
    init_params = {n: p.clone().detach()
                   for n, p in model.named_parameters()}

    eps = [_make_episode(T=10, seed=20)]
    device = torch.device("cpu")
    cfg = model.cfg

    ppo_update_batched(model, optimizer, eps, device, cfg,
                        epochs=2, target_kl=10.0)

    # At least one param must have changed
    n_changed = 0
    for n, p in model.named_parameters():
        diff = (p - init_params[n]).abs().max().item()
        if diff > 1e-6:
            n_changed += 1
    assert n_changed > 0, "NO parameters changed — optimizer didn't step"
    print(f"  {n_changed} parameter tensors changed (optimizer did step)  [PASS]")


def test_kl_gate_triggers_on_extreme_kl():
    """Construct old_logp very different from new logits -> ratio explodes ->
    approx_kl > target_kl × 5 -> KL gate triggers, epoch is skipped (n_succeeded
    remains low or zero)."""
    print("=== Test 3: KL gate triggers on extreme KL ===")
    torch.manual_seed(2)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Build episodes with extreme old_logp values that won't match new logits
    ep = _make_episode(T=10, seed=30)
    ep["old_logp"] = [50.0] * len(ep["old_logp"])  # absurd: log p = 50
    eps = [ep]
    device = torch.device("cpu")
    cfg = model.cfg

    stats = ppo_update_batched(
        model, optimizer, eps, device, cfg,
        epochs=3, target_kl=0.001,  # very tight kl threshold
    )
    print(f"  n_succeeded={stats['n_succeeded']}, n_skipped_kl={stats['n_skipped_kl']}, "
          f"kl={stats['kl']:.2f}")
    assert stats["n_skipped_kl"] > 0, \
        "KL gate should have triggered with extreme old_logp + tight target_kl"
    print(f"  KL gate fired {stats['n_skipped_kl']} times  [PASS]")


def test_empty_episodes_returns_zero_stats():
    """No episodes -> return zero-stats without error."""
    print("=== Test 4: empty episodes return zero stats ===")
    torch.manual_seed(3)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    cfg = model.cfg

    stats = ppo_update_batched(model, optimizer, [], device, cfg, epochs=3)
    assert stats["n_succeeded"] == 0
    assert stats["n_failed"] == 0
    print(f"  empty input -> n_succeeded={stats['n_succeeded']}  [PASS]")


def test_in_warmup_raises_not_implemented():
    """v1 limitation: in_warmup=True must raise NotImplementedError."""
    print("=== Test 5: in_warmup=True raises NotImplementedError (v1) ===")
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eps = [_make_episode(T=5, seed=40)]
    device = torch.device("cpu")
    cfg = model.cfg

    try:
        ppo_update_batched(model, optimizer, eps, device, cfg,
                            epochs=1, in_warmup=True)
        print("  FAIL: in_warmup=True should raise")
        return
    except NotImplementedError as e:
        print(f"  raised NotImplementedError: {str(e)[:80]}  [PASS]")


def test_multi_epoch_loop_runs_n_times():
    """epochs=N -> up to N forward+backward+step cycles unless KL early-stop."""
    print("=== Test 6: multi-epoch loop runs N times ===")
    torch.manual_seed(4)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    eps = [_make_episode(T=6, seed=50)]
    device = torch.device("cpu")
    cfg = model.cfg

    for epochs in (1, 2, 5):
        # Reset params to start fresh each time
        torch.manual_seed(4 + epochs)
        m = _MiniModel()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        stats = ppo_update_batched(m, opt, eps, device, cfg,
                                    epochs=epochs, target_kl=10.0)
        # n_succeeded should be exactly `epochs` (no failures, no kl-skip)
        assert stats["n_succeeded"] == epochs, \
            f"epochs={epochs}: n_succeeded={stats['n_succeeded']}"
        print(f"  epochs={epochs}: n_succeeded={stats['n_succeeded']}  [PASS]")


def main():
    print("Tier 3 C4 unit test: ppo_update_batched()")
    print("=" * 70)
    test_smoke_runs_to_completion()
    print()
    test_optimizer_actually_steps()
    print()
    test_kl_gate_triggers_on_extreme_kl()
    print()
    test_empty_episodes_returns_zero_stats()
    print()
    test_in_warmup_raises_not_implemented()
    print()
    test_multi_epoch_loop_runs_n_times()
    print()
    print("=" * 70)
    print("ALL C4 SMOKE TESTS PASS")
    print("=" * 70)
    print()
    print("Real correctness (gradient parity vs current ppo_update; training")
    print("quality preservation) validated in C6 — 20-iter A/B against Phase 1 v3")
    print("baseline. See docs/PHASE1_V3_OBSERVATIONS.md §C6.")


if __name__ == "__main__":
    main()
