# inference_batcher.py — Async batched GPU inference for concurrent battles.
#
# Extracted from rl_train_v9.py during Session 34 refactor.
# InferenceBatcher collects (features, history) from concurrent async battles,
# batches them into a single GPU forward pass, and dispatches results.

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from arch_compat import (
    call_action_encoder,
    call_policy_logits,
    call_value_logits,
    get_v_support,
)
from model import PokeTransformer
from precision_config import autocast_ctx


class InferenceBatcher:
    """Async batch collector for GPU inference.

    Concurrent battles submit (features, history) and await results.
    When min_batch requests accumulate (or timeout fires), ONE batched
    model.forward() processes them all.
    """

    def __init__(self, model: PokeTransformer, device: torch.device,
                 fp16: bool = False, min_batch: int = 8, timeout_ms: int = 20):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.min_batch = min_batch
        self.timeout_s = timeout_ms / 1000.0
        self._pending: List[Tuple[dict, Optional[torch.Tensor], int, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        # Summary buffer dim = resolved d_temporal (falls back to d_model for legacy checkpoints)
        self._d_model = getattr(model, "d_temporal", model.cfg.d_model)
        # Profiling counters (reset per collection)
        self._prof_batch_sizes = []
        self._prof_gpu_times = []
        self._prof_timeout_fires = 0
        self._prof_normal_fires = 0

    async def submit(self, batch_dict: dict, history: Optional[torch.Tensor],
                     history_len: int) -> dict:
        """Submit inference request. Returns dict with action_logits, value, summary."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((batch_dict, history, history_len, future))

        if len(self._pending) >= self.min_batch:
            # Yield to event loop FIRST so other pending websocket messages
            # can be processed and more battles can submit requests.
            # This dramatically increases batch sizes with many concurrent battles.
            await asyncio.sleep(0)
            self._prof_normal_fires += 1
            await self._fire_batch()
        else:
            # Schedule timeout to fire partial batch
            loop.call_later(self.timeout_s, self._timeout_fire)

        return await future

    def _timeout_fire(self):
        if self._pending:
            self._prof_timeout_fires += 1
            asyncio.ensure_future(self._fire_batch())

    async def _fire_batch(self):
        async with self._lock:
            if not self._pending:
                return
            requests = self._pending[:]
            self._pending.clear()

        try:
            t0 = time.time()
            # Run GPU forward directly on event loop. Blocks for ~14ms but
            # run_in_executor is slower due to GIL contention with CUDA ops.
            results = self._gpu_forward(requests)
            gpu_ms = (time.time() - t0) * 1000
            self._prof_batch_sizes.append(len(requests))
            self._prof_gpu_times.append(gpu_ms)
            for i, (_, _, _, future) in enumerate(requests):
                if not future.done():
                    future.set_result(results[i])
        except Exception as e:
            for _, _, _, future in requests:
                if not future.done():
                    future.set_exception(e)

    def prof_summary(self) -> str:
        """Return profiling summary string and reset counters."""
        if not self._prof_batch_sizes:
            return "no batches"
        sizes = np.array(self._prof_batch_sizes)
        times = np.array(self._prof_gpu_times)
        s = (f"batches={len(sizes)}, size={sizes.mean():.1f}avg/{sizes.min()}-{sizes.max()} "
             f"gpu={times.mean():.1f}ms avg/{times.sum()/1000:.1f}s total "
             f"fires=normal:{self._prof_normal_fires}/timeout:{self._prof_timeout_fires}")
        self._prof_batch_sizes.clear()
        self._prof_gpu_times.clear()
        self._prof_timeout_fires = 0
        self._prof_normal_fires = 0
        return s

    def _gpu_forward(self, requests) -> List[dict]:
        """Fully batched forward: spatial + temporal + heads all batched.

        All N requests are processed in parallel. Variable-length temporal
        histories are left-aligned and padded, with seq_lens for masking.
        """
        N = len(requests)
        model = self.model
        D = self._d_model

        # Stack batch dicts: (1, ...) -> (N, ...)
        mega = self._stack_batches([r[0] for r in requests])

        with torch.no_grad(), autocast_ctx(self.fp16):
            # Phase 1: Batched spatial
            spatial_out, summaries = model.forward_spatial(mega)  # (N, 16, D), (N, D)

            if spatial_out.isnan().any():
                print(f"  [NaN-DIAG] spatial_out has NaN", flush=True)
            if summaries.isnan().any():
                print(f"  [NaN-DIAG] summaries has NaN", flush=True)

            # Phase 2: Batched action encoding (arch-aware — see arch_compat.py)
            action_ctx = call_action_encoder(model, mega, spatial_out)  # (N, 9, D)

            if action_ctx.isnan().any():
                print(f"  [NaN-DIAG] action_ctx has NaN", flush=True)

            # Phase 3: Batched temporal
            # Build left-aligned padded history tensor + current summary
            seq_lens = []
            for i in range(N):
                history = requests[i][1]
                h_len = history.shape[1] if history is not None and history.shape[1] > 0 else 0
                seq_lens.append(h_len + 1)  # +1 for current summary

            max_T = min(max(seq_lens), model.temporal.temporal_context)
            seq_lens_t = torch.tensor(seq_lens, device=self.device, dtype=torch.long).clamp(max=max_T)

            # Pre-allocate padded tensor (zeros for padding)
            all_summaries = torch.zeros(N, max_T, D, device=self.device, dtype=summaries.dtype)

            for i in range(N):
                history = requests[i][1]
                summary_i = summaries[i]  # (D,)

                if history is not None and history.shape[1] > 0:
                    h = history.to(self.device).squeeze(0)  # (T_i, D)
                    # Truncate from left if history + 1 > max_T
                    if h.shape[0] + 1 > max_T:
                        h = h[-(max_T - 1):]
                    h_len = h.shape[0]
                    all_summaries[i, :h_len] = h
                    all_summaries[i, h_len] = summary_i
                else:
                    all_summaries[i, 0] = summary_i

            # FP32 for temporal (matches per-item behavior — summary_attn is fp32)
            temporal_ctx = model.temporal(
                all_summaries.float(), seq_lens_t
            ).to(summaries.dtype)  # (N, D)

            if temporal_ctx.isnan().any():
                nan_mask = temporal_ctx.isnan().any(dim=-1)
                for i in range(N):
                    if nan_mask[i]:
                        print(f"  [NaN-DIAG] temporal has NaN (item {i}, history_len={seq_lens[i]})", flush=True)

            # Phase 4: Batched policy head (arch-aware)
            actor_out = spatial_out[:, 0, :]  # (N, D)
            at = torch.cat([actor_out, temporal_ctx], dim=-1)  # (N, 2D)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)  # (N, 9, 2D)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)  # (N, 9, 3D)
            logits = call_policy_logits(model, pi_input)  # (N, 9)

            if logits.isnan().any():
                print(f"  [NaN-DIAG] policy_head has NaN", flush=True)

            # Apply legal masks
            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

            # Phase 5: Batched value head (arch-aware)
            critic_out = spatial_out[:, 1, :]  # (N, D)
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)  # (N, 2D)
            v_logits = call_value_logits(model, vi)  # (N, 51)
            v_probs = F.softmax(v_logits, dim=-1)
            values = (v_probs * get_v_support(model)).sum(-1)  # (N,)

            # Unpack results
            results = []
            for i in range(N):
                results.append({
                    "action_logits": logits[i],        # (9,)
                    "value": values[i],                 # scalar
                    "summary": summaries[i].float(),    # (D,) always fp32
                })

        return results

    def _stack_batches(self, batch_list: List[dict]) -> dict:
        """Stack N batch dicts (each B=1) into mega-batch (B=N)."""
        mega = {}
        ref = batch_list[0]
        for key in ref:
            if isinstance(ref[key], torch.Tensor):
                mega[key] = torch.cat([b[key] for b in batch_list], dim=0)
            elif isinstance(ref[key], dict):
                mega[key] = {
                    k: torch.cat([b[key][k] for b in batch_list], dim=0)
                    for k in ref[key]
                }
            else:
                mega[key] = ref[key]  # non-tensor, just pass through
        return mega
