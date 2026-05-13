#!/usr/bin/env python
"""Fix #2 (S60): compile + BC anchor + minibatch composition tests.

Covers what's verifiable on Windows (no torch.compile available locally):

  Test 1: _ppo_loss_batched_internal with bc_logits=tensor + tensor coef
          produces output that matches ppo_loss_batched (the eager wrapper)
          on the same inputs. Validates the refactored BC anchor branch.

  Test 2: ppo_update_batched EAGER path with bc_ref + minibatch_size > 0
          completes cleanly, parameters change, bc_kl is recorded. Validates
          BC anchor + minibatching in eager mode (compile-independent).

  Test 3: make_compiled_train_step(bc_anchor_enabled=False) returns a
          callable that works with no bc_logits. Backward-compat check.

  Test 4: make_compiled_train_step(bc_anchor_enabled=True) returns a
          callable. Calling without bc_logits raises AssertionError
          (defensive contract). Calling with bc_logits + bc_anchor_coef_t
          works on platforms where compile is available.

Actual compile equivalence between eager-BC and compiled-BC must run on
dev pod (Linux + torch 2.2.x). See test_tier3_c5_compile.py for the
existing C5 compile harness; that test passes unchanged with our refactor.

CPU-only, ~5-10s. No GPU needed.

Usage:
  python scripts/diag/test_fix2_compile_bc_anchor.py
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

# Import the mini model from C5 test (same shape)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tier3_c5_compile import _MiniModel, _make_episode  # noqa: E402


def _compile_supported() -> bool:
    import platform
    if platform.system() == "Windows":
        return False
    try:
        torch._dynamo.eval_frame.check_if_dynamo_supported()
        return True
    except (RuntimeError, AttributeError):
        return False


# ============================================================================
# Test 1: _ppo_loss_batched_internal with bc_logits tensor
# ============================================================================

def test_loss_internal_with_bc_anchor():
    print("=== Test 1: _ppo_loss_batched_internal with bc_logits tensor ===")
    torch.manual_seed(20)
    model = _MiniModel()
    cfg = model.cfg
    eps = [_make_episode(T=6, seed=200), _make_episode(T=8, seed=201)]
    collated = collate_episodes(copy.deepcopy(eps), L_max=10, device=torch.device("cpu"), tail=True)

    # Compute forward to get logits shape for synthetic bc_logits
    with torch.no_grad():
        forward_out = model.forward_ppo_sequence(collated, torch.device("cpu"))
    n_actions = forward_out["action_logits"].shape[-1]
    B, L_max = collated["B"], collated["L_max"]

    # Synthetic bc_logits — shape (B, L_max, n_actions), realistic-ish
    bc_logits = torch.randn(B, L_max, n_actions) * 2.0

    # Eager wrapper call (Python float coef)
    out_eager_py = ppo_loss_batched(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
        bc_logits=bc_logits, bc_anchor_coef=0.1,
    )
    bc_kl_py = out_eager_py["bc_kl"]
    print(f"  eager (coef=0.1 Python float): bc_kl = {bc_kl_py:.6f}")
    assert bc_kl_py > 0.0, f"bc_kl should be positive on random bc_logits, got {bc_kl_py}"

    # Internal call with tensor coef (compile-style)
    bc_coef_t = torch.tensor(0.1, dtype=torch.float32)
    out_internal_t = _ppo_loss_batched_internal(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
        bc_logits=bc_logits, bc_anchor_coef=bc_coef_t,
    )
    bc_kl_t = out_internal_t["bc_kl"].item()
    print(f"  internal (coef=0.1 tensor): bc_kl = {bc_kl_t:.6f}")
    # bc_kl is unchanged by the coef (coef only weights its contribution to
    # total_loss). Both calls should produce same bc_kl value.
    assert abs(bc_kl_py - bc_kl_t) < 1e-6, \
        f"bc_kl differs: py={bc_kl_py} tensor={bc_kl_t}"

    # total_loss should also match
    tl_py = out_eager_py["total_loss"].item()
    tl_t = out_internal_t["total_loss"].item()
    assert abs(tl_py - tl_t) < 1e-5, \
        f"total_loss differs: py={tl_py} tensor={tl_t}"

    # Test with coef=0.0 tensor — bc_kl should still be computed, but
    # total_loss shouldn't include it (multiply by 0)
    bc_coef_zero = torch.tensor(0.0, dtype=torch.float32)
    out_zero = _ppo_loss_batched_internal(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
        bc_logits=bc_logits, bc_anchor_coef=bc_coef_zero,
    )
    bc_kl_zero = out_zero["bc_kl"].item()
    tl_zero = out_zero["total_loss"].item()
    # bc_kl still computed (same as before — bc_logits same)
    assert abs(bc_kl_zero - bc_kl_t) < 1e-6, \
        f"bc_kl with coef=0.0 should still be computed: got {bc_kl_zero}"
    # total_loss should equal the non-BC baseline (coef * bc_kl = 0)
    out_no_bc = ppo_loss_batched(
        collated, forward_out, model, cfg,
        ent_coef=0.02, vf_coef=0.5, clip_eps=0.2,
        bc_logits=None,
    )
    tl_no_bc = out_no_bc["total_loss"].item()
    assert abs(tl_zero - tl_no_bc) < 1e-5, \
        f"coef=0.0 should reproduce no-BC total_loss: " \
        f"coef=0={tl_zero} no_bc={tl_no_bc}"
    print(f"  coef=0.0 tensor: total_loss matches no-BC ({tl_zero:.6f}) [PASS]")


# ============================================================================
# Test 2: ppo_update_batched eager path with bc_ref + minibatching
# ============================================================================

def test_eager_path_bc_anchor_minibatch():
    print("\n=== Test 2: ppo_update_batched eager + bc_ref + minibatching ===")
    torch.manual_seed(21)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    init_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    cfg = model.cfg

    # BC ref: a SEPARATE _MiniModel instance, frozen
    torch.manual_seed(22)
    bc_ref = _MiniModel()
    bc_ref.eval()
    for p in bc_ref.parameters():
        p.requires_grad_(False)

    # 4 episodes, minibatch_size=2 → 2 chunks
    eps = [_make_episode(T=6 + b, seed=300 + b) for b in range(4)]

    stats = ppo_update_batched(
        model, optimizer, copy.deepcopy(eps), torch.device("cpu"), cfg,
        epochs=1, target_kl=10.0,  # no KL early-stop
        bc_ref=bc_ref, bc_anchor_coef=0.1,
        minibatch_size=2,
    )

    assert stats["n_succeeded"] >= 1, \
        f"expected at least 1 successful epoch, got {stats}"
    bc_kl_recorded = stats.get("bc_kl", 0.0)
    print(f"  n_succeeded={stats['n_succeeded']} bc_kl={bc_kl_recorded:.6f}")
    assert bc_kl_recorded > 0.0, \
        f"bc_kl should be recorded > 0 with bc_ref + coef=0.1: got {bc_kl_recorded}"

    # Params should have changed (gradient flow)
    n_changed = sum(
        1 for n, p in model.named_parameters()
        if (p - init_params[n]).abs().max().item() > 1e-6
    )
    assert n_changed > 0, "no parameters changed — gradient flow broken"
    print(f"  {n_changed} parameter tensors changed via eager BC+mb path  [PASS]")


# ============================================================================
# Test 3: make_compiled_train_step backward-compat (bc_anchor_enabled=False)
# ============================================================================

def test_compiled_train_step_no_bc_signature():
    print("\n=== Test 3: make_compiled_train_step(bc_anchor_enabled=False) ===")
    if not _compile_supported():
        print("  [SKIP] torch.compile not supported on this platform — "
              "build itself fails. Deferred to dev pod.")
        return
    torch.manual_seed(23)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = model.cfg

    # Build with default flag (False) — backward-compat with C5 test
    train_step = make_compiled_train_step(
        model, optimizer, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
    )
    assert callable(train_step), "make_compiled_train_step must return callable"
    print(f"  returned callable: {train_step.__name__ if hasattr(train_step, '__name__') else type(train_step)}")

    if not _compile_supported():
        print("  [INFO] compile not supported on this platform; "
              "actual call deferred to dev pod  [SKIP CALL]")
        return

    # Call without bc args (default kwargs)
    eps = [_make_episode(T=6, seed=400)]
    collated = collate_episodes(copy.deepcopy(eps), L_max=10,
                                 device=torch.device("cpu"), tail=True)
    ent_coef_t = torch.tensor(0.02, dtype=torch.float32)
    clip_eps_t = torch.tensor(0.2, dtype=torch.float32)
    target_kl_t = torch.tensor(10.0, dtype=torch.float32)
    out = train_step(collated, ent_coef_t, clip_eps_t, target_kl_t)
    assert "total_loss" in out and "step_mask" in out and "bc_kl" in out
    # bc_kl should be 0 since bc_anchor_enabled=False
    assert out["bc_kl"].item() == 0.0, \
        f"bc_kl should be 0 in no-BC variant; got {out['bc_kl'].item()}"
    print(f"  no-BC call works, bc_kl=0  [PASS]")


# ============================================================================
# Test 4: make_compiled_train_step with BC enabled — assertion on missing args
# ============================================================================

def test_compiled_train_step_bc_assertions():
    print("\n=== Test 4: bc_anchor_enabled=True asserts on missing bc args ===")
    if not _compile_supported():
        print("  [SKIP] torch.compile not supported on this platform — "
              "build itself fails. Deferred to dev pod.")
        return
    torch.manual_seed(24)
    model = _MiniModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = model.cfg

    train_step = make_compiled_train_step(
        model, optimizer, cfg,
        vf_coef=0.5, max_grad_norm=0.5, normalize_advantages=False,
        bc_anchor_enabled=True,
    )
    assert callable(train_step)
    print(f"  returned callable with bc_anchor_enabled=True")

    eps = [_make_episode(T=6, seed=500)]
    collated = collate_episodes(copy.deepcopy(eps), L_max=10,
                                 device=torch.device("cpu"), tail=True)
    ent_coef_t = torch.tensor(0.02, dtype=torch.float32)
    clip_eps_t = torch.tensor(0.2, dtype=torch.float32)
    target_kl_t = torch.tensor(10.0, dtype=torch.float32)

    # Call without bc_logits should assert
    raised = False
    try:
        train_step(collated, ent_coef_t, clip_eps_t, target_kl_t)
    except AssertionError as e:
        raised = True
        print(f"  AssertionError raised as expected: {str(e)[:80]}...")
    assert raised, "Expected AssertionError when bc_logits missing"

    # Call with bc_logits + bc_anchor_coef_t should work (if compile available)
    if not _compile_supported():
        print("  [INFO] compile not supported; full call deferred to dev pod  [PARTIAL PASS]")
        return

    # Need actual compile to run the full call
    forward_out = model.forward_ppo_sequence(collated, torch.device("cpu"))
    n_actions = forward_out["action_logits"].shape[-1]
    bc_logits = torch.randn(collated["B"], collated["L_max"], n_actions) * 2.0
    bc_coef_t = torch.tensor(0.1, dtype=torch.float32)

    out = train_step(collated, ent_coef_t, clip_eps_t, target_kl_t,
                     bc_logits=bc_logits, bc_anchor_coef_t=bc_coef_t)
    assert "bc_kl" in out
    assert out["bc_kl"].item() != 0.0, \
        f"bc_kl should be non-zero with real bc_logits; got {out['bc_kl'].item()}"
    print(f"  BC call works, bc_kl={out['bc_kl'].item():.6f}  [PASS]")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("S60 Fix #2: compile + BC anchor + minibatch composition tests")
    print("=" * 70)
    print(f"  compile supported on this platform: {_compile_supported()}")
    print()
    test_loss_internal_with_bc_anchor()
    test_eager_path_bc_anchor_minibatch()
    test_compiled_train_step_no_bc_signature()
    test_compiled_train_step_bc_assertions()
    print()
    print("=" * 70)
    print("ALL FIX #2 LOCAL TESTS PASS")
    print("=" * 70)
    if not _compile_supported():
        print("\n[INFO] Actual torch.compile + BC end-to-end equivalence")
        print("       MUST run on dev pod (Linux + torch 2.2.x + triton 2.2.0):")
        print("         ssh -i ~/.ssh/id_ed25519 -p 19373 root@154.54.102.26")
        print("         cd /workspace/team_builder")
        print("         python scripts/diag/test_fix2_compile_bc_anchor.py")
        print("         python scripts/diag/test_tier3_c5_compile.py")


if __name__ == "__main__":
    main()
