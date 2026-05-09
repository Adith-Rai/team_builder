#!/usr/bin/env python
"""Tier 3 C5: torch.compile(train_step) unit tests.

Validates the C5 single-graph compiled train_step (forward + loss + backward
+ clip_grad_norm + optimizer.step in one fused graph via torch.compile +
AOTAutograd).

Acceptance gates:
  1. twohot_target — new F.one_hot impl bit-equivalent to old scatter_ impl
  2. _ppo_loss_batched_internal returns same numeric values as ppo_loss_batched
     (with .item() conversions on approx_kl, ratio_clip_frac, n_valid)
  3. compiled train_step gives same loss + same parameter delta as eager
     baseline on healthy input (CPU fp32 → bit-equivalent)
  4. NaN-injected loss: nan_safe=0, step_mask=0, parameters unchanged
  5. KL gate trips on extreme old_logp/new_logp divergence: kl_safe=0,
     step_mask=0, parameters unchanged
  6. Optimizer step actually changes params on healthy input (gradient flow)

CPU-only, runs in ~10-20s. Real GPU + autocast validation deferred to
dev pod end-to-end smoke (C5 task #3).

Usage:
  python scripts/diag/test_tier3_c5_compile.py
"""

from __future__ import annotations

import copy
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
from ppo import (  # noqa: E402
    _ppo_loss_batched_internal,
    collate_episodes,
    make_compiled_train_step,
    ppo_loss_batched,
    ppo_update_batched,
)


def _compile_supported() -> bool:
    """torch.compile requires Linux/Mac; Windows raises RuntimeError at
    `torch.compile` invocation. Tests 1+2 (twohot, loss internal) don't
    need compile — they validate the prep work locally. Tests 3-6 need
    actual compile and must run on dev pod (port 19373)."""
    import platform
    if platform.system() == "Windows":
        return False
    try:
        torch._dynamo.eval_frame.check_if_dynamo_supported()
        return True
    except (RuntimeError, AttributeError):
        return False


# ----------------------------------------------------------------
# Mini model — mirrors C4's _MiniModel but with F.one_hot twohot_target
# (matches the production model_transformer.py change for compile-cleanliness).
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
    def __init__(self, d_emb: int = 16, n_actions: int = 9, v_bins: int = 51):
        super().__init__()
        self.cfg = _MiniCfg()
        self.value_head = _MiniValueHead(d_in=d_emb, v_bins=v_bins)
        self.policy_head = nn.Linear(d_emb, n_actions)
        self.embed = nn.Embedding(1000, d_emb)
        self._n_actions = n_actions
        self._v_bins = v_bins
        self._d_emb = d_emb
        self.register_buffer(
            "v_support", torch.linspace(self.cfg.v_min, self.cfg.v_max, v_bins)
        )

    def twohot_target(self, value: torch.Tensor) -> torch.Tensor:
        """Compile-friendly broadcast-equality version, mirrors the production
        change in TransformerBattlePolicy.twohot_target. (F.one_hot is NOT
        used because torch 2.2.x dynamic-shape compile fails on it — see
        twohot_target docstring in model_transformer.py.)"""
        v_support = self.value_head.v_support.to(value.device)
        value = value.clamp(self.cfg.v_min, self.cfg.v_max)
        bin_width = v_support[1] - v_support[0]
        idx = (value - v_support[0]) / bin_width
        lo = idx.floor().long().clamp(0, self.cfg.v_bins - 2)
        hi = (lo + 1).clamp(max=self.cfg.v_bins - 1)
        weight_hi = (idx - lo.float()).clamp(0, 1)
        arange = torch.arange(self.cfg.v_bins, device=value.device).unsqueeze(0)
        lo_oh = (arange == lo.unsqueeze(-1)).to(weight_hi.dtype)
        hi_oh = (arange == hi.unsqueeze(-1)).to(weight_hi.dtype)
        return lo_oh * (1 - weight_hi).unsqueeze(-1) + hi_oh * weight_hi.unsqueeze(-1)

    def forward_ppo_sequence(self, collated: dict, device: torch.device) -> dict:
        B = collated["B"]
        L_max = collated["L_max"]
        actions = collated["actions"].to(device)
        pad_mask = collated["pad_mask"].to(device)

        x = self.embed(actions)
        logits = self.policy_head(x)
        v_logits = self.value_head.linear(x)
        value = (F.softmax(v_logits, dim=-1)
                 * self.value_head.v_support).sum(-1)

        logits = logits.clone()
        v_logits = v_logits.clone()
        value = value.clone()
        pad_inv = ~pad_mask
        logits[pad_inv] = -100.0
        v_logits[pad_inv] = 0.0
        value[pad_inv] = 0.0
        return {"action_logits": logits, "v_logits": v_logits, "value": value}


def _twohot_scatter_reference(value: torch.Tensor, cfg, v_support) -> torch.Tensor:
    """Old scatter_-based impl, used as the bit-equivalence reference for
    Test 1. Identical math to the pre-S56 model_transformer.twohot_target."""
    value = value.clamp(cfg.v_min, cfg.v_max)
    bin_width = v_support[1] - v_support[0]
    idx = (value - v_support[0]) / bin_width
    lo = idx.floor().long().clamp(0, cfg.v_bins - 2)
    hi = (lo + 1).clamp(max=cfg.v_bins - 1)
    weight_hi = (idx - lo.float()).clamp(0, 1)
    target = torch.zeros(value.shape[0], cfg.v_bins, device=value.device)
    target.scatter_(1, lo.unsqueeze(1), (1 - weight_hi).unsqueeze(1))
    target.scatter_(1, hi.unsqueeze(1), weight_hi.unsqueeze(1))
    return target


def _make_episode(T: int, A: int = 9, feat_dim: int = 4, seed: int = 0):
    rng = np.random.RandomState(seed)
    feat_batches = [{"x": torch.randn(1, feat_dim)} for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions": list(rng.randint(0, A, size=T).astype(int)),
        "old_logp": list(rng.randn(T).astype(np.float32) * 0.5),
        "advantages": list(rng.randn(T).astype(np.float32)),
        "returns": list(np.tanh(rng.randn(T)).astype(np.float32)),
        "action_masks": [rng.uniform(0.5, 1.0, size=A).astype(np.float32) for _ in range(T)],
    }


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

def test_twohot_one_hot_matches_scatter():
    """New F.one_hot impl must be bit-equivalent to old scatter_ impl
    across the value-target range. Lo and hi never collide (lo clamped
    to [0, v_bins-2], hi=lo+1 clamped to [0, v_bins-1]) so the two
    formulations are mathematically identical."""
    print("=== Test 1: twohot_target F.one_hot == scatter_ (bit-equivalent) ===")
    model = _MiniModel()
    cfg = model.cfg
    v_support = model.value_head.v_support

    # Sweep value range; include boundary, midpoint, NaN-clamp test
    test_values = torch.tensor([
        -1.0, -0.99, -0.5, -0.01, 0.0, 0.01, 0.5, 0.99, 1.0,
        -2.0,  # below min, should clamp
        +2.0,  # above max, should clamp
    ], dtype=torch.float32)

    out_new = model.twohot_target(test_values)
    out_ref = _twohot_scatter_reference(test_values, cfg, v_support)

    max_diff = (out_new - out_ref).abs().max().item()
    assert max_diff == 0.0, \
        f"twohot_target new vs scatter_ diff: {max_diff} (expected 0.0 bit-exact)"
    # Each row should sum to ~1.0 (probability distribution)
    row_sums = out_new.sum(dim=-1)
    sum_diff = (row_sums - 1.0).abs().max().item()
    assert sum_diff < 1e-6, f"row sums not 1.0: {row_sums}"
    print(f"  bit-exact (max diff = {max_diff})  row_sums OK  [PASS]")


def test_loss_internal_matches_eager_wrapper():
    """_ppo_loss_batched_internal returns same numeric values as
    ppo_loss_batched (with .item() conversions on approx_kl/ratio_clip_frac/
    n_valid). Verifies the C5 refactor preserves the eager contract."""
    print("=== Test 2: _ppo_loss_batched_internal vs ppo_loss_batched ===")
    torch.manual_seed(7)
    model = _MiniModel()
    cfg = model.cfg
    eps = [_make_episode(T=8, seed=70), _make_episode(T=12, seed=71)]
    device = torch.device("cpu")
    collated = collate_episodes(eps, device=device)
    forward_out = model.forward_ppo_sequence(collated, device)

    out_internal = _ppo_loss_batched_internal(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
    )
    out_eager = ppo_loss_batched(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
    )

    # Tensor-keys match bit-exact
    for k in ("total_loss", "pi_loss", "entropy", "v_loss"):
        diff = (out_internal[k] - out_eager[k]).abs().item()
        assert diff == 0.0, f"{k}: internal vs eager diff {diff}"
    # Scalar-keys: internal returns tensor, eager returns Python float — same value
    for k in ("approx_kl", "ratio_clip_frac"):
        v_internal = out_internal[k].item()
        v_eager = out_eager[k]
        diff = abs(v_internal - v_eager)
        assert diff < 1e-7, f"{k}: internal {v_internal} vs eager {v_eager}"
    n_valid_int = int(out_internal["n_valid"].item())
    n_valid_eager = out_eager["n_valid"]
    assert n_valid_int == n_valid_eager, f"n_valid: {n_valid_int} vs {n_valid_eager}"
    print(f"  all loss components bit-exact  [PASS]")


def test_compiled_train_step_matches_eager():
    """Compiled train_step must give same loss values + same parameter
    delta as the eager ppo_update_batched path on healthy input.

    CPU fp32 — bit-equivalent expected (no autocast).

    NOTE: torch.compile on CPU may fall back to eager execution; this test
    validates correctness of the call graph (same output regardless of
    whether actual graph fusion happens). Real fusion measured on dev pod.
    """
    print("=== Test 3: compiled train_step == eager (CPU, fp32) ===")
    torch.manual_seed(11)
    eps = [_make_episode(T=8, seed=110), _make_episode(T=10, seed=111)]
    device = torch.device("cpu")

    # ---- Eager run ----
    torch.manual_seed(11)
    model_e = _MiniModel()
    opt_e = torch.optim.AdamW(model_e.parameters(), lr=1e-3)
    init_params_e = {n: p.clone().detach() for n, p in model_e.named_parameters()}
    cfg = model_e.cfg
    stats_e = ppo_update_batched(
        model_e, opt_e, copy.deepcopy(eps), device, cfg,
        epochs=1, target_kl=10.0,  # no KL early-stop
    )
    final_params_e = {n: p.clone().detach() for n, p in model_e.named_parameters()}

    # ---- Compiled run ----
    torch.manual_seed(11)
    model_c = _MiniModel()
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
    compiled_step = make_compiled_train_step(
        model_c, opt_c, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
    )
    stats_c = ppo_update_batched(
        model_c, opt_c, copy.deepcopy(eps), device, cfg,
        epochs=1, target_kl=10.0,
        compiled_step=compiled_step,
    )

    # Both must complete same number of epochs
    assert stats_e["n_succeeded"] == stats_c["n_succeeded"], \
        f"n_succeeded mismatch eager={stats_e['n_succeeded']} compiled={stats_c['n_succeeded']}"

    # Loss values must agree (small fp tolerance — random.shuffle order can
    # differ via separate random streams; but with epochs=1 + same input
    # they should match closely)
    for k in ("pi", "v", "ent", "kl"):
        diff = abs(stats_e[k] - stats_c[k])
        # Eager and compiled both call random.shuffle(episodes) in the same
        # order since we re-seed both runs identically. Bit-exact in fp32.
        assert diff < 1e-4, \
            f"{k}: eager {stats_e[k]} vs compiled {stats_c[k]} diff {diff}"

    # Parameter deltas must agree
    max_param_diff = 0.0
    for n in init_params_e:
        delta_e = (final_params_e[n] - init_params_e[n])
        p_c = dict(model_c.named_parameters())[n]
        delta_c = (p_c - init_params_e[n]).detach()
        diff = (delta_e - delta_c).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
    # CPU fp32, same seed, same shuffle order → expect very small diff
    # (allow 1e-4 for any nondeterminism in fused AdamW or compile reordering)
    assert max_param_diff < 1e-4, \
        f"parameter delta diff between eager + compiled: {max_param_diff}"
    print(f"  loss + param delta match (max param diff = {max_param_diff:.2e})  [PASS]")


def test_compiled_optimizer_actually_steps():
    """Compiled train_step must actually update parameters on healthy input.
    Smoke-checks that backward + optimizer.step in the compiled graph fire."""
    print("=== Test 4: compiled train_step changes params (gradient flow) ===")
    torch.manual_seed(12)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)  # large lr
    init_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    cfg = model.cfg
    eps = [_make_episode(T=10, seed=120)]
    device = torch.device("cpu")

    compiled_step = make_compiled_train_step(
        model, optimizer, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
    )
    ppo_update_batched(
        model, optimizer, eps, device, cfg,
        epochs=2, target_kl=10.0,
        compiled_step=compiled_step,
    )

    n_changed = sum(
        1 for n, p in model.named_parameters()
        if (p - init_params[n]).abs().max().item() > 1e-6
    )
    assert n_changed > 0, "compiled train_step didn't change any parameters"
    print(f"  {n_changed} parameter tensors changed via compiled path  [PASS]")


def test_compiled_kl_gate_skips_step():
    """Extreme old_logp → ratio explodes → approx_kl > target_kl × 5 →
    in-graph kl_safe mask is 0 → masked loss is 0 → backward yields zero
    grads → optimizer.step is a no-op for the affected params.

    Verifies: kl_safe in returned mask is 0; parameters do NOT change."""
    print("=== Test 5: compiled KL gate masks step (params unchanged) ===")
    torch.manual_seed(13)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
    init_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    cfg = model.cfg
    ep = _make_episode(T=10, seed=130)
    ep["old_logp"] = [50.0] * len(ep["old_logp"])  # extreme — KL will explode
    eps = [ep]
    device = torch.device("cpu")

    compiled_step = make_compiled_train_step(
        model, optimizer, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
    )
    stats = ppo_update_batched(
        model, optimizer, eps, device, cfg,
        epochs=3, target_kl=0.001,  # tight threshold so gate fires
        compiled_step=compiled_step,
    )

    # KL gate must have fired
    assert stats["n_skipped_kl"] > 0, \
        f"expected KL skips, got n_skipped_kl={stats['n_skipped_kl']}"

    # Parameters should be UNCHANGED: every step was masked.
    # (AdamW momentum buffers DO update slightly even with zero grads, but
    # the parameter values themselves move only via gradient × lr × step_mask
    # = 0. Allow a tiny tolerance for any optimizer state initialization.)
    max_param_diff = max(
        (p - init_params[n]).abs().max().item()
        for n, p in model.named_parameters()
    )
    assert max_param_diff < 1e-5, \
        f"params changed despite KL gate firing: max diff {max_param_diff}"
    print(f"  KL gate fired {stats['n_skipped_kl']}× — params unchanged "
          f"(max diff {max_param_diff:.2e})  [PASS]")


def test_compiled_nan_gate_skips_step():
    """NaN-injected advantage → NaN propagates to loss → in-graph nan_safe
    mask is 0 → masked loss replaced with 0 via nan_to_num → backward
    yields zero grads → optimizer.step is a no-op.

    Verifies: nan_safe in returned mask is 0; parameters do NOT change.

    NOTE: build_ppo_episodes' caller is responsible for upstream NaN checks;
    this test verifies the compiled train_step's in-graph safety net."""
    print("=== Test 6: compiled NaN gate masks step (params unchanged) ===")
    torch.manual_seed(14)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
    init_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    cfg = model.cfg
    ep = _make_episode(T=10, seed=140)
    # Inject NaN into advantages → propagates through pi_loss → total_loss = NaN
    ep["advantages"] = [float("nan")] + list(ep["advantages"])[1:]
    eps = [ep]
    device = torch.device("cpu")

    compiled_step = make_compiled_train_step(
        model, optimizer, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
    )
    stats = ppo_update_batched(
        model, optimizer, eps, device, cfg,
        epochs=2, target_kl=10.0,
        compiled_step=compiled_step,
    )

    assert stats["n_skipped_nan"] > 0, \
        f"expected NaN skips, got n_skipped_nan={stats['n_skipped_nan']}"

    max_param_diff = max(
        (p - init_params[n]).abs().max().item()
        for n, p in model.named_parameters()
    )
    assert max_param_diff < 1e-5, \
        f"params changed despite NaN gate firing: max diff {max_param_diff}"
    print(f"  NaN gate fired {stats['n_skipped_nan']}× — params unchanged "
          f"(max diff {max_param_diff:.2e})  [PASS]")


def main():
    print("Tier 3 C5 unit test: torch.compile(train_step)")
    print("=" * 70)
    # Tier-1 tests (no compile needed) run everywhere
    test_twohot_one_hot_matches_scatter()
    print()
    test_loss_internal_matches_eager_wrapper()
    print()

    # Tier-2 tests (compile needed) skip on Windows
    if not _compile_supported():
        print("=" * 70)
        print("[INFO] torch.compile not supported on this platform "
              "(Windows or torch lacks dynamo).")
        print("       Tests 1+2 PASS locally (prep validation).")
        print("       Tests 3-6 (compile equivalence + safety masks) MUST run")
        print("       on dev pod — Linux + torch 2.2.x + triton 2.2.0:")
        print("         ssh -p 19373 root@154.54.102.26")
        print("         cd /workspace/team_builder && python "
              "scripts/diag/test_tier3_c5_compile.py")
        print("=" * 70)
        return

    test_compiled_train_step_matches_eager()
    print()
    test_compiled_optimizer_actually_steps()
    print()
    test_compiled_kl_gate_skips_step()
    print()
    test_compiled_nan_gate_skips_step()
    print()
    print("=" * 70)
    print("ALL C5 UNIT TESTS PASS")
    print("=" * 70)
    print()
    print("Real fusion measurement (compile cache hit, wall-time speedup,")
    print("autocast bf16/fp16) deferred to dev pod end-to-end smoke.")


if __name__ == "__main__":
    main()
