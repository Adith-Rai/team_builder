"""S64 Phase B.2 bit-equivalence gate: TemporalTransformer.forward vs forward_packed.

LOAD-BEARING gate for B.2. We MUST get this to pass with BC v10 weights at
fp32 eval mode before any downstream work. If it fails:
- Within ~1e-4 at fp32: probably acceptable epsilon from 4-layer transformer
  compose; investigate but proceed with widened tolerance.
- > 1e-4: real bug. Q/K/V split, scaling, mask predicate, or pos-embed indexing.

What this tests:
1. Synthetic episodes (varied T, all <= cfg.temporal_context).
2. Same per-episode summary tensors packed into BOTH legacy (B, L_max, D)
   and packed (sum_T, D) layouts.
3. Run model.temporal.forward(legacy_summaries, seq_lens, return_all_positions=True).
4. Run model.temporal.forward_packed(packed_summaries, cu_seqlens).
5. Assert allclose at every valid (b, t) ↔ idx position.

Run on prod (flex_attention requires torch 2.5+):
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python test_temporal_packed.py
"""
import sys
import torch

# Ensure src is on path
sys.path.insert(0, '.')

from ppo import load_checkpoint

# Tolerance band rationale: flex_attention (Triton-kernel) and the
# nn.MultiheadAttention slow path (cuDNN/aten math) compute the same
# attention but cannot be bit-identical at fp32 — different kernels, different
# accumulation orders. With BC v10 weights through 4 temporal layers, observed
# max_abs grows from ~2e-6 at T=1 (fp32 noise floor — math is correct) up to
# ~7e-6 at T=200 (legitimate compose drift). rtol=1e-4 atol=1e-5 catches real
# semantic bugs (which produce ~1e-3 or worse — verified empirically by the
# earlier fast-path-vs-slow-path failure at max_abs ~2e-3) while accommodating
# unavoidable cross-kernel reorder. bf16 would widen further (1e-2).
#
# Note on max_rel reporting: max_rel is informational, NOT a pass criterion.
# allclose uses |a-b| <= atol + rtol*|b|. When legacy outputs are near zero
# (typical for some transformer output positions), max_rel blows up at tiny
# absolute diffs — misleading.
FP32_RTOL = 1e-4
FP32_ATOL = 1e-5


def _build_paired_summaries(T_list, D, device, dtype=torch.float32, seed=0):
    """Build (legacy_padded, packed, cu_seqlens, seq_lens) from the SAME
    per-episode random summaries. Both formats hold identical data; legacy
    has zero-padding at non-valid positions, packed has no padding."""
    torch.manual_seed(seed)
    B = len(T_list)
    L_max = max(T_list)
    sum_T = sum(T_list)

    per_ep = [torch.randn(T, D, device=device, dtype=dtype) for T in T_list]

    # Packed: cat along dim=0 → (sum_T, D)
    packed = torch.cat(per_ep, dim=0)

    # cu_seqlens: int32 [0, T_0, T_0+T_1, ..., sum_T] — Phase A convention
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(T_list, dtype=torch.int32, device=device).cumsum(0)

    # seq_lens: long (B,) for legacy forward
    seq_lens = torch.tensor(T_list, dtype=torch.long, device=device)

    # Legacy padded: (B, L_max, D), zeros at padding positions
    legacy = torch.zeros(B, L_max, D, device=device, dtype=dtype)
    for b, (T, summ) in enumerate(zip(T_list, per_ep)):
        legacy[b, :T] = summ

    return legacy, packed, cu, seq_lens


def _check_equiv(out_legacy, out_packed, T_list, cu_seqlens, label, rtol, atol):
    """Compare legacy (B, L_max, D) vs packed (sum_T, D) at all valid positions."""
    B = len(T_list)
    cu = cu_seqlens.long()
    max_abs_diff_overall = 0.0
    max_rel_diff_overall = 0.0
    all_pass = True
    for b in range(B):
        start = int(cu[b].item())
        stop = int(cu[b + 1].item())
        legacy_slice = out_legacy[b, :T_list[b]]  # (T_b, D)
        packed_slice = out_packed[start:stop]      # (T_b, D)
        diff = (legacy_slice - packed_slice).abs()
        max_abs = diff.max().item()
        denom = legacy_slice.abs().clamp(min=1e-12)
        max_rel = (diff / denom).max().item()
        max_abs_diff_overall = max(max_abs_diff_overall, max_abs)
        max_rel_diff_overall = max(max_rel_diff_overall, max_rel)
        passed = torch.allclose(legacy_slice, packed_slice, rtol=rtol, atol=atol)
        marker = "OK" if passed else "FAIL"
        print(f"  [{label}] ep {b}: T={T_list[b]}, max_abs={max_abs:.2e}, max_rel={max_rel:.2e}  [{marker}]")
        if not passed:
            all_pass = False
    print(f"  [{label}] overall: max_abs={max_abs_diff_overall:.2e}, "
          f"max_rel={max_rel_diff_overall:.2e}, rtol={rtol}, atol={atol}")
    return all_pass, max_abs_diff_overall, max_rel_diff_overall


def test_temporal_packed_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    # Disable nn.TransformerEncoderLayer's fast path so legacy goes through
    # _sa_block + _ff_block (which is what _packed_layer_forward mirrors).
    # The fast path (`torch._transformer_encoder_layer_fwd`, a fused C++
    # kernel) is enabled in eval mode + batch_first + norm_first and produces
    # numerically-near-but-not-identical results due to operation fusion.
    # Comparing fast-path-legacy vs slow-path-packed gives misleading drift
    # at ~1e-3 even when the math is correct. Disabling fast path makes the
    # bit-equiv comparison apples-to-apples.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
        print(f"mha fastpath disabled: {not torch.backends.mha.get_fastpath_enabled()}")

    print("Loading BC v10...")
    model, cfg, _ = load_checkpoint("data/models/bc/v10_padded_for_cis_dev.pt", device)
    model.eval()  # CRITICAL: dropout must be identity for bit-equiv
    print(f"BC v10 loaded. d_temporal={cfg.d_temporal}, n_heads={cfg.n_heads}, "
          f"n_temporal_layers={cfg.n_temporal_layers}, dropout={cfg.dropout}, "
          f"temporal_context={cfg.temporal_context}")
    print()

    D = cfg.d_temporal

    # ---- Case 1: B=1 single episode (the simplest case; isolates any
    # core math bug before introducing multi-episode boundary handling)
    print("Case 1: B=1 single episode")
    T_list_1 = [50]
    legacy_1, packed_1, cu_1, seq_1 = _build_paired_summaries(T_list_1, D, device, seed=0)
    with torch.no_grad():
        out_legacy_1 = model.temporal(legacy_1, seq_1, return_all_positions=True)
        out_packed_1 = model.temporal.forward_packed(packed_1, cu_1)
    pass_1, abs_1, rel_1 = _check_equiv(
        out_legacy_1, out_packed_1, T_list_1, cu_1,
        label="B=1", rtol=FP32_RTOL, atol=FP32_ATOL,
    )
    print()

    # ---- Case 2: B=3 multi-episode varied lengths (exercises cu_seqlens
    # episode-boundary handling in pos_in_ep + BlockMask)
    print("Case 2: B=3 varied lengths")
    T_list_2 = [50, 100, 75]
    legacy_2, packed_2, cu_2, seq_2 = _build_paired_summaries(T_list_2, D, device, seed=1)
    with torch.no_grad():
        out_legacy_2 = model.temporal(legacy_2, seq_2, return_all_positions=True)
        out_packed_2 = model.temporal.forward_packed(packed_2, cu_2)
    pass_2, abs_2, rel_2 = _check_equiv(
        out_legacy_2, out_packed_2, T_list_2, cu_2,
        label="B=3", rtol=FP32_RTOL, atol=FP32_ATOL,
    )
    print()

    # ---- Case 3: B=5 hitting the temporal_context bound (max T allowed)
    print("Case 3: B=5 lengths near temporal_context")
    tc = cfg.temporal_context
    T_list_3 = [tc, tc // 2, 1, tc - 1, tc // 4]
    legacy_3, packed_3, cu_3, seq_3 = _build_paired_summaries(T_list_3, D, device, seed=2)
    with torch.no_grad():
        out_legacy_3 = model.temporal(legacy_3, seq_3, return_all_positions=True)
        out_packed_3 = model.temporal.forward_packed(packed_3, cu_3)
    pass_3, abs_3, rel_3 = _check_equiv(
        out_legacy_3, out_packed_3, T_list_3, cu_3,
        label="B=5/edge", rtol=FP32_RTOL, atol=FP32_ATOL,
    )
    print()

    all_pass = pass_1 and pass_2 and pass_3
    if all_pass:
        print("=" * 60)
        print("B.2 GATE: PASS")
        print(f"  Max abs diff across all 3 cases: {max(abs_1, abs_2, abs_3):.2e}")
        print(f"  Max rel diff across all 3 cases: {max(rel_1, rel_2, rel_3):.2e}")
        print("  TemporalTransformer.forward_packed bit-equivalent to forward at fp32 eval.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("B.2 GATE: FAIL")
        print("  Diagnose: B=1 alone failing → core MHA/FFN math bug.")
        print("           B=1 OK + B=3 failing → cu_seqlens/BlockMask boundary bug.")
        print("           B=1,B=3 OK + B=5 failing → temporal_context edge case.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    test_temporal_packed_equivalence()
