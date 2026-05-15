"""S64 Phase B.6 end-to-end smoke: ppo_update_batched(packed=True).

Wires up everything from B.2-B.5 + the new --packed branch in
ppo_update_batched. Builds synthetic episodes, runs ONE epoch of
ppo_update_batched with packed=True, and verifies:
  - completes without error (no shape mismatch, no NaN/inf, no KeyError)
  - returns stats dict with all expected keys
  - n_succeeded > 0 (at least one epoch landed an optimizer.step)
  - loss values are finite

Also runs the same input through packed=False as a sanity comparison —
both should complete; numerical equivalence is already gated by B.5.

Run on prod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python test_packed_update_smoke.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from ppo import load_checkpoint, ppo_update_batched
from test_forward_ppo_sequence_packed import _make_synthetic_episode


EXPECTED_STAT_KEYS = {
    "pi", "v", "ent", "kl", "ratio_clip_frac",
    "value_mean", "return_mean", "adv_abs_mean", "bc_kl",
    "n_succeeded", "n_failed", "n_skipped_kl", "n_skipped_nan",
}


def _check_stats(stats: dict, label: str) -> bool:
    """Verify stats dict structure + finite values."""
    all_pass = True
    missing = EXPECTED_STAT_KEYS - set(stats.keys())
    if missing:
        print(f"  [{label}] MISSING keys: {missing}")
        all_pass = False

    if stats.get("n_succeeded", 0) == 0:
        print(f"  [{label}] FAIL: n_succeeded=0 — all epochs skipped/failed "
              f"(n_failed={stats.get('n_failed')} "
              f"n_skipped_kl={stats.get('n_skipped_kl')} "
              f"n_skipped_nan={stats.get('n_skipped_nan')})")
        all_pass = False

    for k in ["pi", "v", "ent", "kl", "value_mean", "return_mean",
              "adv_abs_mean", "bc_kl"]:
        if k not in stats:
            continue
        v = float(stats[k])
        finite = (v == v) and (v != float("inf")) and (v != float("-inf"))
        if not finite:
            print(f"  [{label}] FAIL: stats[{k}]={v} not finite")
            all_pass = False

    print(f"  [{label}] n_succeeded={stats['n_succeeded']}, "
          f"n_failed={stats['n_failed']}, "
          f"n_skipped_kl={stats['n_skipped_kl']}, "
          f"n_skipped_nan={stats['n_skipped_nan']}")
    print(f"  [{label}] pi={stats['pi']:.4f}, v={stats['v']:.4f}, "
          f"ent={stats['ent']:.4f}, kl={stats['kl']:.4f}, "
          f"bc_kl={stats['bc_kl']:.4f}")
    return all_pass


def test_packed_update_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    print("Loading BC v10...")
    model, cfg, _ = load_checkpoint("data/models/bc/v10_padded_for_cis_dev.pt", device)
    print(f"BC v10 loaded.")
    print()

    # Build synthetic episodes — small but realistic. Mirrors B.5 fixtures.
    T_list = [10, 30, 20, 15, 25]
    episodes = [_make_synthetic_episode(T, A=cfg.format_config.n_actions, seed=i)
                for i, T in enumerate(T_list)]

    A = cfg.format_config.n_actions
    print(f"Synthesized {len(T_list)} episodes, T_list={T_list}, "
          f"total transitions={sum(T_list)}, n_actions={A}")
    print()

    # BC ref = the same loaded model (for the smoke we just need bc_anchor
    # to fire through the eager path — not testing the BC training-dynamic
    # correctness here).
    bc_ref = model

    # Common kwargs for both runs
    common = dict(
        epochs=1,
        clip_eps=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        normalize_advantages=False,
        compiled_step=None,
        bc_ref=bc_ref,
        bc_anchor_coef=0.1,
        minibatch_size=2,  # chunks of 2 → 3 chunks for 5 episodes
    )

    all_pass = True

    # ---- legacy path (sanity that we didn't break it) ----
    print("Run 1: ppo_update_batched(packed=False) — legacy path")
    # Fresh optimizer (each run gets its own to keep state independent)
    optim_legacy = torch.optim.AdamW(model.parameters(), lr=1e-5)
    stats_legacy = ppo_update_batched(
        model, optim_legacy, list(episodes), device, cfg,
        packed=False, **common,
    )
    if not _check_stats(stats_legacy, "legacy"):
        all_pass = False
    print()

    # ---- packed path (B.6 wiring) ----
    print("Run 2: ppo_update_batched(packed=True) — packed path (B.6)")
    optim_packed = torch.optim.AdamW(model.parameters(), lr=1e-5)
    stats_packed = ppo_update_batched(
        model, optim_packed, list(episodes), device, cfg,
        packed=True, **common,
    )
    if not _check_stats(stats_packed, "packed"):
        all_pass = False
    print()

    # ---- Cross-path sanity: stats are SIMILAR (not bit-equiv — different
    # optimizer.step states + autocast bf16 noise from gradient flow). Just
    # confirm they're in the same ballpark (within 30%). Real numerical
    # equivalence at the loss level is gated by B.5.
    print("Cross-path sanity check (stats within ballpark):")
    for k in ["pi", "v", "ent", "kl"]:
        l = float(stats_legacy[k])
        p = float(stats_packed[k])
        rel = abs(l - p) / max(abs(l), abs(p), 1e-6)
        marker = "OK" if rel < 0.3 else "NOTE"
        print(f"  {k:8s}: legacy={l:.6e}  packed={p:.6e}  rel_diff={rel:.2%}  [{marker}]")
    print()

    # ---- compiled+packed rejection check ----
    print("Run 3: --packed + --compile must raise NotImplementedError")
    try:
        # Pass a dummy non-None compiled_step
        ppo_update_batched(
            model, optim_packed, list(episodes), device, cfg,
            packed=True,
            compiled_step=lambda *a, **kw: None,  # not actually called
            **{k: v for k, v in common.items() if k != "compiled_step"},
        )
        print("  FAIL: expected NotImplementedError")
        all_pass = False
    except NotImplementedError as e:
        print(f"  OK: {e}")
    print()

    print("=" * 60)
    if all_pass:
        print("B.6 GATE: PASS")
        print("  --packed wiring is correct end-to-end:")
        print("    - ppo_update_batched(packed=True) completes without error")
        print("    - All stat keys present, all values finite")
        print("    - Legacy path still works (no regression)")
        print("    - --packed + --compile raises NotImplementedError as designed")
        print("=" * 60)
        sys.exit(0)
    else:
        print("B.6 GATE: FAIL — diagnose before proceeding to B.7")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    test_packed_update_smoke()
