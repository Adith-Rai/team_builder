"""mp_centralized_collect.py - Centralized Inference Server (CIS) for cloud PPO.

PHASE 1 SKELETON (Session 51, 2026-05-07): single CIS process owns the GPU
model; workers serialize obs to numpy and send via mp.Pipe; CIS deserializes,
runs forward, returns numpy logits. No batching, no production worker
integration yet - just the scaffolding + logits identity test gate.

See `docs/CENTRALIZED_INFERENCE_DESIGN.md` for full design rationale.
Phase progression:
  1. Skeleton + logits identity (THIS FILE): one-shot single-request path
  2. Batching across workers + N=4 sustained
  3. Weight sync from main + low-priority CUDA stream
  4. Re-enable mp+pipeline overlap, production validation

Why CIS at all (recap):
  Production --mp gives each worker its own GPU model copy. Works fine for
  --mp alone, but --mp --pipeline silently no-ops due to GPU contention
  deadlock between worker forwards and main's optimizer.step. Centralizing
  inference into one GPU process - using CUDA stream priority to arbitrate
  vs main's optimizer step on a HIGH-priority stream - resolves the
  contention. CIS forwards land on a LOW-priority stream and always make
  progress, just slower during update windows. No deadlock.

Design decisions (per CENTRALIZED_INFERENCE_DESIGN.md §design):
  A. Routing flag: new --cis flag (TBD; --mp keeps routing to mp_disk_collect
     until CIS is fully validated; both paths coexist during transition).
  B. Weight sync: file-based atomic write (Phase 3); reuses mp_disk's pattern.
  C. IPC: mp.Pipe per-worker (FD-only, no SemLock - bypasses the resource_tracker
     race that fires at N>=4 with mp.Queue on RunPod containers).
  D. Tensor type: numpy ONLY across IPC. No torch tensor IPC - that's the
     diagnosed root cause of mmap explosion in mp_collect_v2's design. Workers
     do `feat_dict_torch -> numpy_dict -> Pipe.send`; CIS does
     `numpy_dict -> torch_dict (on GPU) -> forward -> logits.cpu().numpy() ->
     Pipe.send`. Pickle CPU overhead at our scale (~75K req/iter) is ~3-5 ms
     total, far less than the iter-time saving from real pipeline overlap.
  E. CUDA stream priority: Phase 3 work. Main on default (priority=0, high);
     CIS on torch.cuda.Stream(priority=-1) (low). Without it, both compete
     unmanaged.
  F. Pipeline overlap re-enable: Phase 4 work in train_rl.py
     _start_background_collection.

POKE_LOOP threading note (cookbook §3j): Phase 2+ workers create poke-env
Players in their own asyncio loops. ANY sequential _cancel_listener pattern
in CIS-worker code MUST yield wall-clock time afterward (await
asyncio.sleep(1.5)) - same fix as in mp_disk_collect's _play_vs_opp.
mp_disk's first leak-fix attempt hung at 99% CPU because it skipped this;
don't repeat that mistake here.
"""

from __future__ import annotations

import multiprocessing as mp_stdlib
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp_torch

# Defensive: even though we don't send torch tensors across IPC, set the
# strategy in case a legacy code path leaks one.
try:
    mp_torch.set_sharing_strategy('file_system')
except Exception:
    pass


# spawn context chosen over forkserver - same rationale as mp_disk_collect.py:
# CPython 3.11 multiprocessing.resource_tracker race at N>=4 with forkserver,
# spawn fully isolates child Python state.
def _get_mp_ctx():
    try:
        return mp_torch.get_context('spawn')
    except (ValueError, OSError):
        return mp_torch.get_context('fork')


# =============================
# Numpy <-> torch dict serializers
# =============================

def torch_dict_to_numpy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a feat_dict (matching unpack_turn_batch's schema) to numpy.

    Handles nested dicts (field_banks, transition_ids, active_move_banks)
    and tolerates non-tensor values (passed through unchanged).

    The returned dict is pickle-safe and mp.Pipe-safe (no torch tensors).
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # CPU + numpy. Detach in case any tensor has grad attached.
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            out[k] = {k2: (v2.detach().cpu().numpy()
                           if isinstance(v2, torch.Tensor) else v2)
                      for k2, v2 in v.items()}
        else:
            out[k] = v
    return out


def numpy_dict_to_torch(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Inverse of torch_dict_to_numpy: numpy back to torch tensors on device.

    Preserves nested-dict shape. Tensors are placed on `device`.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, dict):
            out[k] = {k2: (torch.from_numpy(v2).to(device)
                           if isinstance(v2, np.ndarray) else v2)
                      for k2, v2 in v.items()}
        else:
            out[k] = v
    return out


def _output_to_numpy(out: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Pack the model.forward()'s return dict (action_logits, value, v_logits,
    summary, spatial_output) into numpy for IPC. Spatial_output is sometimes
    not needed downstream; keep it for now but consumers may drop it."""
    return {
        "action_logits":   out["action_logits"].detach().float().cpu().numpy(),
        "value":           out["value"].detach().float().cpu().numpy(),
        "v_logits":        out["v_logits"].detach().float().cpu().numpy(),
        "summary":         out["summary"].detach().float().cpu().numpy(),
        # spatial_output is large (B, 222, 256); only ship if a worker really
        # needs it. Phase 1 doesn't, so omit to halve IPC payload.
        # "spatial_output": out["spatial_output"].detach().float().cpu().numpy(),
    }


# =============================
# CIS process entrypoint
# =============================

def _cis_main(req_pipe, resp_pipe, ckpt_path: str, device_str: str,
              fp16: bool = True, amp_dtype_name: Optional[str] = None) -> None:
    """CIS subprocess entrypoint. Loads model from ckpt, services inference
    requests until shutdown.

    Protocol:
      Inbound on req_pipe:
        {"cmd": "infer", "batch": numpy_dict}        -> {"status": "ok", "out": numpy_dict}
        {"cmd": "ping"}                              -> {"status": "ok"}
        {"cmd": "shutdown"}                          -> exits cleanly

      On error:                                       -> {"status": "error",
                                                          "exc_msg": str,
                                                          "traceback": str}

    Args:
      req_pipe:   mp.Pipe end the parent writes to and we read from
      resp_pipe:  mp.Pipe end we write responses to and parent reads from
      ckpt_path:  initial weights path (Phase 1 - Phase 3 adds reload)
      device_str: 'cuda' typically
      fp16:       legacy autocast bool (matches existing fp16-bool plumbing)
      amp_dtype_name: 'fp16' | 'bf16' | 'fp32' | None - sets precision_config
                      global for autocast_ctx in this child process
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Set precision_config global on this side (subprocess has its own globals)
    try:
        from precision_config import set_amp_dtype, parse_amp_dtype
        set_amp_dtype(parse_amp_dtype(amp_dtype_name))
    except Exception:
        pass

    from precision_config import autocast_ctx
    from ppo import load_checkpoint

    device = torch.device(device_str)

    try:
        model, cfg, _ = load_checkpoint(ckpt_path, device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[CIS] loaded {ckpt_path} ({n_params/1e6:.1f}M params, device={device})",
              flush=True)
    except Exception as e:
        # Can't recover from load failure. Notify parent + exit.
        try:
            resp_pipe.send({
                "status": "fatal",
                "exc_msg": f"CIS failed to load ckpt: {e}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        return

    # Notify ready
    try:
        resp_pipe.send({"status": "ready"})
    except Exception:
        return

    # Service loop
    while True:
        try:
            req = req_pipe.recv()
        except (EOFError, BrokenPipeError) as e:
            print(f"[CIS] req_pipe closed: {e}", flush=True)
            break

        cmd = req.get("cmd") if isinstance(req, dict) else None

        if cmd == "shutdown":
            print("[CIS] shutdown received", flush=True)
            break

        if cmd == "ping":
            try:
                resp_pipe.send({"status": "ok"})
            except Exception:
                break
            continue

        if cmd == "infer":
            try:
                np_batch = req["batch"]
                torch_batch = numpy_dict_to_torch(np_batch, device)

                # Optional: history (B, T, d_temporal) if the worker passed any.
                # Phase 1 doesn't ship history through IPC; the model.forward
                # handles history=None just fine for single-turn forward.
                history = req.get("history")
                history_lens = req.get("history_lens")
                if history is not None and isinstance(history, np.ndarray):
                    history = torch.from_numpy(history).to(device)
                if history_lens is not None and isinstance(history_lens, np.ndarray):
                    history_lens = torch.from_numpy(history_lens).to(device)

                with torch.no_grad(), autocast_ctx(fp16):
                    out = model(torch_batch, history=history, history_lens=history_lens)

                np_out = _output_to_numpy(out)
                resp_pipe.send({"status": "ok", "out": np_out})
            except Exception as e:
                tb = traceback.format_exc()
                try:
                    resp_pipe.send({
                        "status": "error",
                        "exc_msg": str(e),
                        "traceback": tb,
                    })
                except Exception:
                    break
            continue

        # Unknown command
        try:
            resp_pipe.send({
                "status": "error",
                "exc_msg": f"unknown cmd: {cmd!r}",
                "traceback": "",
            })
        except Exception:
            break


# =============================
# Main-side handle for managing the CIS process
# =============================

class CISClient:
    """Main-side handle to a single CIS process. Owns the request pipe and
    blocks on responses for synchronous inference.

    Phase 1 use: spawn(), infer(numpy_batch) -> numpy_out, shutdown().
    Phase 2 will add concurrent multi-worker request multiplexing.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 fp16: bool = True, amp_dtype_name: Optional[str] = None):
        self.ckpt_path = ckpt_path
        self.device = device
        self.fp16 = fp16
        self.amp_dtype_name = amp_dtype_name
        self._proc = None
        self._req_writer = None
        self._resp_reader = None

    def spawn(self, ready_timeout_s: float = 60.0) -> None:
        """Start CIS subprocess. Blocks until CIS reports ready."""
        ctx = _get_mp_ctx()
        # ctrl direction (parent -> CIS): parent writes, CIS reads
        ctrl_reader, ctrl_writer = ctx.Pipe(duplex=False)
        # response direction (CIS -> parent): CIS writes, parent reads
        resp_reader, resp_writer = ctx.Pipe(duplex=False)

        self._proc = ctx.Process(
            target=_cis_main,
            args=(ctrl_reader, resp_writer,
                  self.ckpt_path, self.device, self.fp16, self.amp_dtype_name),
            daemon=False,
        )
        self._proc.start()
        # Close child ends in parent
        ctrl_reader.close()
        resp_writer.close()
        self._req_writer = ctrl_writer
        self._resp_reader = resp_reader

        # Wait for ready
        if not self._resp_reader.poll(timeout=ready_timeout_s):
            raise RuntimeError(f"CIS did not signal ready within {ready_timeout_s}s")
        msg = self._resp_reader.recv()
        if msg.get("status") == "fatal":
            raise RuntimeError(f"CIS fatal: {msg.get('exc_msg')}\n{msg.get('traceback', '')}")
        if msg.get("status") != "ready":
            raise RuntimeError(f"CIS unexpected ready msg: {msg!r}")

    def infer(self, numpy_batch: Dict[str, Any],
              history: Optional[np.ndarray] = None,
              history_lens: Optional[np.ndarray] = None,
              timeout_s: float = 30.0) -> Dict[str, np.ndarray]:
        """Send a single inference request. Blocks until response arrives.

        Phase 1 sync API. Phase 2 will add async/batched.
        """
        if self._req_writer is None:
            raise RuntimeError("CIS not spawned")
        msg = {"cmd": "infer", "batch": numpy_batch}
        if history is not None:
            msg["history"] = history
        if history_lens is not None:
            msg["history_lens"] = history_lens
        self._req_writer.send(msg)

        if not self._resp_reader.poll(timeout=timeout_s):
            raise TimeoutError(f"CIS infer response not received within {timeout_s}s")
        resp = self._resp_reader.recv()
        status = resp.get("status")
        if status == "ok":
            return resp["out"]
        if status == "error":
            raise RuntimeError(f"CIS infer error: {resp.get('exc_msg')}\n"
                               f"{resp.get('traceback', '')}")
        raise RuntimeError(f"CIS unexpected response: {resp!r}")

    def ping(self, timeout_s: float = 5.0) -> bool:
        if self._req_writer is None:
            return False
        try:
            self._req_writer.send({"cmd": "ping"})
            if not self._resp_reader.poll(timeout=timeout_s):
                return False
            resp = self._resp_reader.recv()
            return resp.get("status") == "ok"
        except Exception:
            return False

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        try:
            if self._req_writer is not None:
                self._req_writer.send({"cmd": "shutdown"})
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.kill()
        try:
            if self._req_writer is not None:
                self._req_writer.close()
            if self._resp_reader is not None:
                self._resp_reader.close()
        except Exception:
            pass
        self._proc = None
        self._req_writer = None
        self._resp_reader = None


# =============================
# Phase 2: multi-worker CIS with cross-worker batching
# =============================

def _stack_numpy_batches(batches: list) -> Dict[str, np.ndarray]:
    """Concatenate a list of feat_dict numpy batches along the batch axis.

    Each input batch is a dict matching unpack_turn_batch's schema (with
    nested dicts for field_banks/transition_ids/active_move_banks). All
    batches must have matching schema; only the leading dimension differs.

    Returns a single mega-batch dict, plus the per-input batch sizes for
    later un-stacking."""
    out = {}
    ref = batches[0]
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            out[k] = np.concatenate([b[k] for b in batches], axis=0)
        elif isinstance(v, dict):
            out[k] = {k2: np.concatenate([b[k][k2] for b in batches], axis=0)
                      for k2 in v}
        else:
            out[k] = v
    return out


def _slice_numpy_output(np_out: Dict[str, np.ndarray],
                       start: int, end: int) -> Dict[str, np.ndarray]:
    """Slice a CIS output dict (action_logits, value, v_logits, summary) along
    batch axis [start:end]. Used to dispatch per-request results back to each
    worker after a batched forward."""
    return {k: v[start:end] for k, v in np_out.items()}


def _cis_main_multi(worker_req_readers, worker_resp_writers,
                    ckpt_path: str, device_str: str,
                    fp16: bool = True,
                    amp_dtype_name: Optional[str] = None,
                    min_batch: int = 8,
                    timeout_ms: int = 15) -> None:
    """Phase 2 CIS subprocess entrypoint. Multiplexes inference requests across
    N worker pipes via mp.connection.wait, accumulates pending requests until
    min_batch OR timeout_ms, fires one batched forward, dispatches per-request
    responses back via per-worker response pipes.

    Protocol per pipe:
      Inbound on req_pipe[i]:  same as Phase 1 single-CIS protocol
      Outbound on resp_pipe[i]: per-request response; for batched infer we
                                slice the mega-batch result and send the
                                appropriate slice to each worker.

    Args:
      worker_req_readers:  list of mp.Pipe reader ends, one per worker
      worker_resp_writers: list of mp.Pipe writer ends, one per worker
      (other args: same as _cis_main)
    """
    import warnings
    from multiprocessing.connection import wait as mp_wait
    warnings.filterwarnings("ignore")

    try:
        from precision_config import set_amp_dtype, parse_amp_dtype
        set_amp_dtype(parse_amp_dtype(amp_dtype_name))
    except Exception:
        pass

    from precision_config import autocast_ctx
    from ppo import load_checkpoint

    device = torch.device(device_str)

    # Ready signal sent on EVERY worker resp pipe so each handle's spawn()
    # call can poll its own pipe without reading another worker's signal.
    n_workers = len(worker_req_readers)
    assert len(worker_resp_writers) == n_workers

    # Map pipe object -> worker index for fast lookup after mp_wait
    pipe_to_widx = {p: i for i, p in enumerate(worker_req_readers)}

    try:
        model, cfg, _ = load_checkpoint(ckpt_path, device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[CIS-multi N={n_workers}] loaded {ckpt_path} "
              f"({n_params/1e6:.1f}M params, device={device}, "
              f"min_batch={min_batch}, timeout_ms={timeout_ms})", flush=True)
    except Exception as e:
        for w in worker_resp_writers:
            try:
                w.send({"status": "fatal",
                        "exc_msg": f"CIS failed to load ckpt: {e}",
                        "traceback": traceback.format_exc()})
            except Exception:
                pass
        return

    # Send ready to each worker's resp pipe so each handle gets a clean signal.
    for w in worker_resp_writers:
        try:
            w.send({"status": "ready"})
        except Exception:
            pass

    # Service loop with cross-worker batching.
    # pending: list of 5-tuples (worker_idx, req_id_or_None, batch_size,
    #                            np_batch, np_history_or_None)
    pending: list = []
    last_fire_t = time.time()
    timeout_s = timeout_ms / 1000.0
    closed_workers: set = set()

    # Lazy-import arch_compat helpers; only needed if any request includes
    # a history (i.e., production call path uses the InferenceBatcher-style
    # forward with batched temporal). Phase 1 callers without history fall
    # back to model.forward(mega) which handles the no-history case.
    try:
        from arch_compat import call_action_encoder, call_policy_logits, call_value_logits, get_v_support
        import torch.nn.functional as F
        _have_arch_compat = True
    except Exception:
        _have_arch_compat = False

    d_temporal = getattr(model, "d_temporal", model.cfg.d_model)

    def _fire_batch():
        """Fire one batched forward over `pending` and dispatch responses.

        Two paths:
        - If NO request in pending has a history, use the simple
          model(mega_batch) path (Phase 1 single-call style). All requests
          get processed at once, history-free.
        - If ANY request has a history, use the InferenceBatcher pattern:
          forward_spatial(mega) -> action_encoder + padded temporal +
          policy_head + value_head. This mirrors inference_batcher.py
          _gpu_forward exactly so worker-side V9RLPlayer behavior matches
          the local-InferenceBatcher path bit-for-bit.

        Per-request output keys: action_logits (9), value (scalar),
        v_logits (51), summary (d_temporal). `summary` is what workers
        accumulate into their per-battle history buffer.
        """
        nonlocal last_fire_t
        if not pending:
            return
        try:
            # Are any requests carrying a non-empty history?
            any_history = any(p[4] is not None and p[4].shape[1] > 0
                              for p in pending)

            stacked = _stack_numpy_batches([p[3] for p in pending])
            torch_batch = numpy_dict_to_torch(stacked, device)

            with torch.no_grad(), autocast_ctx(fp16):
                if not any_history or not _have_arch_compat:
                    # Simple path: history-free or arch_compat unavailable.
                    out = model(torch_batch)
                    mega_np = _output_to_numpy(out)
                else:
                    # Full batched-with-histories path. Mirrors
                    # inference_batcher._gpu_forward.
                    N = len(pending)
                    spatial_out, summaries = model.forward_spatial(torch_batch)
                    action_ctx = call_action_encoder(model, torch_batch, spatial_out)

                    # Build padded all_summaries with per-request histories.
                    # Per-request bsize is always 1 in worker submits (one
                    # battle's turn at a time); but defensively support B>1.
                    seq_lens = []
                    histories_torch = []
                    cursor = 0
                    for (widx, req_id, bsize, _, np_hist) in pending:
                        # np_hist is (B, T_i, D) numpy or None. For our
                        # worker submits B=bsize. We expand each B-slot's
                        # row independently (different histories may differ
                        # within a batch; in practice bsize=1 so only one
                        # history per request).
                        if np_hist is not None and np_hist.shape[1] > 0:
                            h = torch.from_numpy(np_hist).to(device)
                        else:
                            h = None
                        for b in range(bsize):
                            if h is not None and h.shape[1] > 0:
                                h_b = h[b] if h.shape[0] > 1 else h.squeeze(0)
                                # h_b: (T_i, D)
                                seq_lens.append(int(h_b.shape[0]) + 1)
                                histories_torch.append(h_b)
                            else:
                                seq_lens.append(1)
                                histories_torch.append(None)
                        cursor += bsize

                    total_B = cursor  # == sum of bsizes == summaries.shape[0]
                    max_T = min(max(seq_lens), model.temporal.temporal_context)
                    seq_lens_t = torch.tensor(seq_lens, device=device,
                                              dtype=torch.long).clamp(max=max_T)
                    all_summaries = torch.zeros(
                        total_B, max_T, d_temporal, device=device,
                        dtype=summaries.dtype)
                    for i in range(total_B):
                        h_i = histories_torch[i]
                        if h_i is not None and h_i.shape[0] > 0:
                            if h_i.shape[0] + 1 > max_T:
                                h_i = h_i[-(max_T - 1):]
                            h_len = h_i.shape[0]
                            all_summaries[i, :h_len] = h_i
                            all_summaries[i, h_len] = summaries[i]
                        else:
                            all_summaries[i, 0] = summaries[i]

                    temporal_ctx = model.temporal(
                        all_summaries.float(), seq_lens_t
                    ).to(summaries.dtype)

                    actor_out = spatial_out[:, 0, :]
                    at = torch.cat([actor_out, temporal_ctx], dim=-1)
                    at_exp = at.unsqueeze(1).expand(-1, 9, -1)
                    pi_input = torch.cat([at_exp, action_ctx], dim=-1)
                    logits = call_policy_logits(model, pi_input)

                    if "legal_mask" in torch_batch:
                        logits = logits.float().masked_fill(
                            torch_batch["legal_mask"] < 0.5, -100.0)

                    critic_out = spatial_out[:, 1, :]
                    vi = torch.cat([critic_out, temporal_ctx], dim=-1)
                    v_logits = call_value_logits(model, vi)
                    v_probs = F.softmax(v_logits, dim=-1)
                    values = (v_probs * get_v_support(model)).sum(-1)

                    mega_np = {
                        "action_logits": logits.detach().float().cpu().numpy(),
                        "value":         values.detach().float().cpu().numpy(),
                        "v_logits":      v_logits.detach().float().cpu().numpy(),
                        "summary":       summaries.detach().float().cpu().numpy(),
                    }

            # Dispatch per-request slices
            cursor = 0
            for entry in pending:
                widx = entry[0]
                req_id = entry[1]
                bsize = entry[2]
                end = cursor + bsize
                slice_out = _slice_numpy_output(mega_np, cursor, end)
                cursor = end
                msg = {"status": "ok", "out": slice_out}
                if req_id is not None:
                    msg["req_id"] = req_id
                try:
                    worker_resp_writers[widx].send(msg)
                except Exception:
                    closed_workers.add(widx)
        except Exception as e:
            tb = traceback.format_exc()
            err = {"status": "error", "exc_msg": str(e), "traceback": tb}
            for entry in pending:
                widx = entry[0]
                req_id = entry[1]
                msg = dict(err)
                if req_id is not None:
                    msg["req_id"] = req_id
                try:
                    worker_resp_writers[widx].send(msg)
                except Exception:
                    closed_workers.add(widx)

        pending.clear()
        last_fire_t = time.time()

    while True:
        # Active pipes = readers we still listen to (skip closed workers)
        active_pipes = [p for p in worker_req_readers
                        if pipe_to_widx[p] not in closed_workers]
        if not active_pipes:
            print(f"[CIS-multi] all worker pipes closed, exiting", flush=True)
            break

        # Compute remaining timeout budget for this batch window
        elapsed = time.time() - last_fire_t
        remaining = max(0.0, timeout_s - elapsed) if pending else timeout_s
        # If pending is empty, block longer (no urgency); else cap at remaining
        wait_to = remaining if pending else timeout_s

        ready = mp_wait(active_pipes, timeout=wait_to)

        for r in ready:
            widx = pipe_to_widx[r]
            try:
                req = r.recv()
            except (EOFError, BrokenPipeError):
                closed_workers.add(widx)
                continue

            cmd = req.get("cmd") if isinstance(req, dict) else None

            if cmd == "shutdown":
                closed_workers.add(widx)
                continue

            if cmd == "ping":
                try:
                    worker_resp_writers[widx].send({"status": "ok"})
                except Exception:
                    closed_workers.add(widx)
                continue

            if cmd == "reload":
                # Phase 3: hot-reload weights from disk. Main writes weights
                # atomically to a path (same pattern as mp_disk_collect_sync's
                # _save_worker_weights_atomic), sends "reload" with the path,
                # and CIS does load_state_dict in-place. The model architecture
                # is unchanged - only the parameters update. Returns "ok" once
                # load completes so main knows it's safe to start the next
                # iter's collect.
                try:
                    weights_path = req.get("weights_path")
                    if not weights_path:
                        raise ValueError("reload cmd missing weights_path")
                    state = torch.load(weights_path, map_location=device,
                                       weights_only=True)
                    sd = state.get("model_state_dict") if isinstance(state, dict) else state
                    if sd is None:
                        raise ValueError(f"no model_state_dict at {weights_path}")
                    # Mirror load_checkpoint's _orig_mod strip: if the state_dict
                    # was saved from a torch.compile-wrapped model, keys carry
                    # an extra "._orig_mod." segment that the unwrapped model
                    # doesn't know about. Without this, strict=False would
                    # silently skip every wrapped key and leave the model in
                    # a half-perturbed state. ppo.py:397 has the canonical
                    # pattern.
                    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
                    # strict=True so key mismatches surface loudly. Half-loaded
                    # models silently produce wrong outputs, which is exactly
                    # the bug Phase 3 Stage 6 caught before this fix shipped.
                    missing, unexpected = [], []
                    try:
                        model.load_state_dict(sd, strict=True)
                    except RuntimeError as load_err:
                        # If strict load fails, fall back to non-strict but
                        # surface the error so caller knows reload was partial.
                        result = model.load_state_dict(sd, strict=False)
                        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
                        print(f"[CIS] WARN reload had key mismatches: {load_err}",
                              flush=True)
                    del state, sd
                    import gc
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    msg = {"status": "ok",
                           "missing_keys": missing[:5],
                           "unexpected_keys": unexpected[:5]}
                    worker_resp_writers[widx].send(msg)
                except Exception as e:
                    tb = traceback.format_exc()
                    try:
                        worker_resp_writers[widx].send({
                            "status": "error",
                            "exc_msg": f"reload failed: {e}",
                            "traceback": tb,
                        })
                    except Exception:
                        closed_workers.add(widx)
                continue

            if cmd == "infer":
                np_batch = req["batch"]
                # Determine batch size from first ndarray-shaped field
                bsize = None
                for k, v in np_batch.items():
                    if isinstance(v, np.ndarray):
                        bsize = v.shape[0]
                        break
                    if isinstance(v, dict):
                        for v2 in v.values():
                            if isinstance(v2, np.ndarray):
                                bsize = v2.shape[0]
                                break
                        if bsize is not None:
                            break
                if bsize is None:
                    bsize = 1
                # History (B, T_i, D) carried as numpy. None if first turn.
                np_history = req.get("history")
                pending.append((widx, req.get("req_id"), bsize, np_batch, np_history))

        # Fire if accumulated enough OR timed out with non-empty pending
        accumulated = sum(p[2] for p in pending)
        timed_out = (time.time() - last_fire_t) >= timeout_s
        if pending and (accumulated >= min_batch or timed_out):
            _fire_batch()


class CISClientHandle:
    """Per-worker view of a CIS server. Holds one (req_writer, resp_reader)
    pipe pair that routes through a single shared CIS subprocess. Multiple
    CISClientHandles can call .infer() concurrently from different threads
    or processes; CIS multiplexes via mp.connection.wait."""

    def __init__(self, req_writer, resp_reader, worker_idx: int):
        self.req_writer = req_writer
        self.resp_reader = resp_reader
        self.worker_idx = worker_idx

    def infer(self, numpy_batch: Dict[str, Any], timeout_s: float = 30.0,
              req_id: Optional[int] = None) -> Dict[str, np.ndarray]:
        msg = {"cmd": "infer", "batch": numpy_batch}
        if req_id is not None:
            msg["req_id"] = req_id
        self.req_writer.send(msg)
        if not self.resp_reader.poll(timeout=timeout_s):
            raise TimeoutError(f"CIS infer (worker {self.worker_idx}) "
                               f"response not received within {timeout_s}s")
        resp = self.resp_reader.recv()
        status = resp.get("status")
        if status == "ok":
            return resp["out"]
        if status == "error":
            raise RuntimeError(f"CIS infer error (worker {self.worker_idx}): "
                               f"{resp.get('exc_msg')}\n{resp.get('traceback','')}")
        raise RuntimeError(f"CIS unexpected response: {resp!r}")

    def ping(self, timeout_s: float = 5.0) -> bool:
        try:
            self.req_writer.send({"cmd": "ping"})
            if not self.resp_reader.poll(timeout=timeout_s):
                return False
            return self.resp_reader.recv().get("status") == "ok"
        except Exception:
            return False

    def reload(self, weights_path: str, timeout_s: float = 60.0) -> Dict[str, Any]:
        """Phase 3: signal CIS to load fresh weights from disk. Blocks until
        CIS confirms load complete. Returns the response dict (with optional
        missing_keys/unexpected_keys for diagnostics)."""
        self.req_writer.send({"cmd": "reload", "weights_path": weights_path})
        if not self.resp_reader.poll(timeout=timeout_s):
            raise TimeoutError(f"CIS reload (worker {self.worker_idx}) "
                               f"response not received within {timeout_s}s")
        resp = self.resp_reader.recv()
        if resp.get("status") == "ok":
            return resp
        raise RuntimeError(f"CIS reload error: {resp.get('exc_msg')}\n"
                           f"{resp.get('traceback','')}")

    def shutdown(self) -> None:
        try:
            self.req_writer.send({"cmd": "shutdown"})
        except Exception:
            pass


class CISServer:
    """Phase 2 server: spawns one CIS subprocess that multiplexes N worker pipes
    with cross-worker batching. Returns N CISClientHandle objects, one per
    worker, that can be used independently from threads/processes."""

    def __init__(self, ckpt_path: str, n_workers: int, device: str = "cuda",
                 fp16: bool = True, amp_dtype_name: Optional[str] = None,
                 min_batch: int = 8, timeout_ms: int = 15):
        self.ckpt_path = ckpt_path
        self.n_workers = n_workers
        self.device = device
        self.fp16 = fp16
        self.amp_dtype_name = amp_dtype_name
        self.min_batch = min_batch
        self.timeout_ms = timeout_ms
        self._proc = None
        self._handles: list = []

    def spawn(self, ready_timeout_s: float = 60.0) -> list:
        """Start CIS subprocess. Returns list of N CISClientHandle objects."""
        ctx = _get_mp_ctx()

        worker_req_readers = []   # child-side reader for ctrl direction
        worker_req_writers = []   # parent-side writer for ctrl direction
        worker_resp_readers = []  # parent-side reader for resp direction
        worker_resp_writers = []  # child-side writer for resp direction

        for _ in range(self.n_workers):
            ctrl_r, ctrl_w = ctx.Pipe(duplex=False)
            resp_r, resp_w = ctx.Pipe(duplex=False)
            worker_req_readers.append(ctrl_r)
            worker_req_writers.append(ctrl_w)
            worker_resp_readers.append(resp_r)
            worker_resp_writers.append(resp_w)

        self._proc = ctx.Process(
            target=_cis_main_multi,
            args=(worker_req_readers, worker_resp_writers,
                  self.ckpt_path, self.device, self.fp16, self.amp_dtype_name,
                  self.min_batch, self.timeout_ms),
            daemon=False,
        )
        self._proc.start()
        # Close the child-side pipe ends in the parent (held only by child)
        for r in worker_req_readers:
            r.close()
        for w in worker_resp_writers:
            w.close()

        # Build handles + wait for ready on each
        self._handles = []
        for i in range(self.n_workers):
            self._handles.append(CISClientHandle(
                req_writer=worker_req_writers[i],
                resp_reader=worker_resp_readers[i],
                worker_idx=i,
            ))

        # Block until each handle's resp pipe receives "ready"
        for h in self._handles:
            if not h.resp_reader.poll(timeout=ready_timeout_s):
                raise RuntimeError(f"CIS worker {h.worker_idx} not ready in "
                                   f"{ready_timeout_s}s")
            msg = h.resp_reader.recv()
            if msg.get("status") == "fatal":
                raise RuntimeError(f"CIS fatal: {msg.get('exc_msg')}\n"
                                   f"{msg.get('traceback','')}")
            if msg.get("status") != "ready":
                raise RuntimeError(f"CIS unexpected ready msg: {msg!r}")

        return self._handles

    def reload_weights(self, weights_path: str, timeout_s: float = 60.0) -> Dict[str, Any]:
        """Phase 3 weight sync: signal CIS to reload from disk. Routes through
        worker 0's handle (any handle works - one CIS process serves all).
        Production caller writes weights atomically first (via
        _save_worker_weights_atomic in mp_disk_collect.py or equivalent), then
        invokes this. Blocks until reload completes; safe to start next iter's
        infer requests after this returns."""
        if not self._handles:
            raise RuntimeError("CIS not spawned")
        return self._handles[0].reload(weights_path, timeout_s=timeout_s)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        for h in self._handles:
            h.shutdown()
        if self._proc.is_alive():
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.kill()
        for h in self._handles:
            try:
                h.req_writer.close()
                h.resp_reader.close()
            except Exception:
                pass
        self._proc = None
        self._handles = []


# =============================
# Phase 4: CISInferenceBatcher - drop-in for InferenceBatcher in workers
# =============================

class CISInferenceBatcher:
    """Drop-in replacement for `inference_batcher.InferenceBatcher` for use
    inside CIS workers (which don't own a local GPU model).

    Same `submit(batch, history, h_len)` async API as InferenceBatcher;
    same return-dict shape (action_logits, value, summary as torch tensors
    on `device`). Internally, each submit serializes the batch+history to
    numpy, sends via the CISClientHandle, awaits the numpy response, and
    converts back to torch tensors.

    Design choice: NO local batching in the worker. Each concurrent battle's
    submit call goes individually to CIS over the worker's pipe. CIS does
    ALL batching across all workers (cross-worker batching at min_batch=8
    OR 15ms timeout), which is the architectural advantage of CIS over the
    per-worker-batcher mp_disk path. With 8 workers x ~100 concurrent
    battles each = ~800 in-flight inference requests at peak; CIS sees
    them as one stream and batches naturally.

    IPC cost: numpy serialization adds ~1-2ms per submit. mp.Pipe.send is
    blocking-but-fast (no SemLock, FD-only). The worker's asyncio loop
    yields between submits anyway, so total throughput is gated by CIS's
    GPU forward rate, not the per-call IPC overhead.

    Compatible API: V9RLPlayer constructs `InferenceBatcher(...)` and calls
    `await batcher.submit(...)` - we substitute this class with the same
    signature, V9RLPlayer is unchanged.
    """

    def __init__(self, handle: "CISClientHandle", device: torch.device,
                 fp16: bool = False, min_batch: int = 8, timeout_ms: int = 20):
        # `min_batch` and `timeout_ms` are accepted for InferenceBatcher API
        # compat but not used here - CIS does the batching, not us.
        self.handle = handle
        self.device = device
        self.fp16 = fp16
        # Profiling counters (reset per collection); same names as
        # InferenceBatcher so any stats-printing code keeps working.
        self._prof_batch_sizes: list = []
        self._prof_gpu_times: list = []
        self._prof_timeout_fires = 0
        self._prof_normal_fires = 0
        self._prof_total_requests = 0

    async def submit(self, batch_dict: dict, history: Optional[torch.Tensor],
                     history_len: int) -> dict:
        """Send one inference request to CIS. Returns dict with
        `action_logits`, `value`, `summary` as torch tensors on self.device,
        single-item (no leading batch dim) - matching InferenceBatcher's
        per-request return shape.
        """
        import asyncio

        self._prof_total_requests += 1
        t0 = time.time()

        # Serialize batch to numpy (B=1 since this is one battle's turn).
        np_batch = torch_dict_to_numpy(batch_dict)

        # History: (1, T, D) torch -> numpy. CIS reconstructs.
        np_history = None
        if history is not None and history.shape[1] > 0:
            np_history = history.detach().cpu().numpy()

        # mp.Pipe.send blocks the asyncio loop. Run it in a thread executor
        # so other concurrent battle coroutines keep making progress.
        loop = asyncio.get_running_loop()
        np_out = await loop.run_in_executor(
            None,
            lambda: self.handle.infer(np_batch, history=np_history, timeout_s=60.0),
        )

        # Convert back to torch on device. Squeeze leading B=1 dim to match
        # InferenceBatcher's per-request return shape: action_logits (9,),
        # value scalar, summary (D,).
        action_logits = torch.from_numpy(np_out["action_logits"]).to(self.device).squeeze(0)
        value = torch.from_numpy(np_out["value"]).to(self.device).squeeze(0)
        summary = torch.from_numpy(np_out["summary"]).to(self.device).squeeze(0)

        gpu_ms = (time.time() - t0) * 1000
        self._prof_batch_sizes.append(1)
        self._prof_gpu_times.append(gpu_ms)
        self._prof_normal_fires += 1

        return {
            "action_logits": action_logits,
            "value": value,
            "summary": summary,
        }

    def prof_summary(self) -> str:
        """Same return-format as InferenceBatcher for log compat."""
        if not self._prof_batch_sizes:
            return "no requests"
        sizes = np.array(self._prof_batch_sizes)
        times = np.array(self._prof_gpu_times)
        s = (f"requests={len(sizes)}, "
             f"per-call ms={times.mean():.1f}avg/{times.sum()/1000:.1f}s total")
        self._prof_batch_sizes.clear()
        self._prof_gpu_times.clear()
        self._prof_total_requests = 0
        self._prof_normal_fires = 0
        return s


# =============================
# Phase 4: production worker entrypoint + sync collect API
# =============================
#
# The production CIS worker mirrors mp_disk_collect.py's worker but with
# two key differences:
#   1. NO local model load - CIS holds the only model. Workers receive a
#      CISClientHandle in their spawn args and use CISInferenceBatcher
#      (above) for all inference.
#   2. NO opp model load either - opp inference also routes through CIS.
#      Workers just track which opp ckpt is active for PFSP weighting; CIS
#      reloads to that ckpt before running self-play vs that opp.
#
# Trade-off: CIS sees TWO model state versions per iter (main player +
# active opp). One option: CIS holds two models, swaps via worker hint.
# Cleaner for Phase 4 minimum: single model. Workers pause between opps
# for CIS to reload. This is OK because each opp gets its own batch of
# games (n_per_opp=100-150 typically); reload overhead amortizes.
#
# Phase 4.1 (this commit): worker entrypoint + sync collect function +
# CUDA stream priority. Train_rl.py wiring deferred to Phase 4.2.
# Sustained validation deferred to Phase 4.3.


def _cis_worker_main(worker_id: int, ctrl_pipe, result_pipe,
                     cis_req_writer, cis_resp_reader) -> None:
    """CIS-aware worker process entrypoint. Stays alive across iters,
    reloads CISClientHandle's pipe references on respawn.

    Receives via ctrl_pipe:
      {"cmd": "collect_iter", iter_n, n_games, max_concurrent, server_url,
       opp_pool, fp16, rs_cfg, turn_cap, battle_format,
       procedural_teams_path, device, opponent_device, rng_seed, amp_dtype}
      {"cmd": "shutdown"}

    Posts via result_pipe (same shape as mp_disk_collect):
      {"status": "done"|"error"|"heartbeat", worker_id, iter_n, traj_path,
       n_games_played, wins, losses, ties, ...}

    The cis_req_writer / cis_resp_reader are pipe ends to the CIS server
    process. Worker reconstructs a CISClientHandle from these and passes
    to CISInferenceBatcher.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import torch.multiprocessing as _mp_child
        _mp_child.set_sharing_strategy('file_system')
    except Exception:
        pass

    import logging
    logging.basicConfig(level=logging.WARNING)

    # Reconstruct handle from inherited pipe ends.
    cis_handle = CISClientHandle(
        req_writer=cis_req_writer,
        resp_reader=cis_resp_reader,
        worker_idx=worker_id,
    )

    # Worker-local state survives across iters.
    state = {
        "device": None,
        "cis_handle": cis_handle,
    }

    def _send_heartbeat(iter_n: int, n_done: int, n_total: int):
        try:
            result_pipe.send({
                "status": "heartbeat", "worker_id": worker_id,
                "iter_n": iter_n, "n_games_done": n_done,
                "n_games_total": n_total, "ts": time.time(),
            })
        except (BrokenPipeError, OSError):
            pass

    while True:
        try:
            if not ctrl_pipe.poll(timeout=30.0):
                _send_heartbeat(iter_n=-1, n_done=0, n_total=0)
                continue
            cmd = ctrl_pipe.recv()
        except (EOFError, BrokenPipeError):
            break
        except Exception as e:
            print(f"[cis-worker {worker_id}] ctrl_pipe.recv failed: {e}", flush=True)
            break

        if cmd.get("cmd") == "shutdown":
            break
        if cmd.get("cmd") != "collect_iter":
            continue

        iter_n = cmd["iter_n"]
        _send_heartbeat(iter_n=iter_n, n_done=0, n_total=cmd.get("n_games", 0))
        try:
            _do_collect_iter_cis(state, worker_id, cmd, result_pipe, _send_heartbeat)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                result_pipe.send({
                    "status": "error", "worker_id": worker_id,
                    "iter_n": iter_n,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e), "traceback": tb,
                })
            except Exception:
                pass
            print(f"[cis-worker {worker_id}] iter {iter_n} crashed:\n{tb}", flush=True)
            return


def _do_collect_iter_cis(state, worker_id, cmd, result_pipe, heartbeat_fn):
    """One iter of CIS-routed collect. Reuses mp_disk's _run_collect_in_worker
    structure, but injects a CISInferenceBatcher instead of constructing a
    local-model InferenceBatcher.

    Note: opp inference in this Phase 4.1 cut runs through CIS too. The CIS
    server holds ONE model at a time (the main player's weights). For
    self-play, CIS would need to reload between main player and opp -
    expensive. Phase 4.1 punts this by always using the SAME model for
    self-play (i.e., main player vs itself). Real PFSP per-opp weights
    require Phase 4.2 (CIS holds two model slots and swaps).
    """
    import gzip
    import pickle as _pickle
    import os as _os

    iter_n = cmd["iter_n"]
    n_games = cmd["n_games"]
    max_concurrent = cmd["max_concurrent"]
    server_url = cmd["server_url"]
    rs_cfg = cmd["rs_cfg"]
    fp16 = cmd.get("fp16", True)
    turn_cap = cmd.get("turn_cap", 300)
    battle_format = cmd.get("battle_format", "gen9ou")
    procedural_teams_path = cmd.get("procedural_teams_path")
    rng_seed = cmd.get("rng_seed", 0)
    opp_pool = cmd["opp_pool"]
    temp_range = tuple(cmd.get("temp_range", (1.0, 2.25)))
    opp_temp_range = tuple(cmd.get("opp_temp_range", temp_range))
    device_str = cmd.get("device", "cuda")

    # amp_dtype propagation (same pattern as mp_disk_collect).
    try:
        from precision_config import set_amp_dtype, parse_amp_dtype
        set_amp_dtype(parse_amp_dtype(cmd.get("amp_dtype")))
    except Exception:
        pass

    import random as _random
    _random.seed(rng_seed + worker_id * 1000 + iter_n)

    device = torch.device(device_str)
    state["device"] = device

    # Liveness probe thread (same pattern as mp_disk_collect's, includes
    # heartbeat-from-thread mitigation - decoupled from asyncio loop).
    import threading
    _liveness_state = {"alive": True, "n_done": 0, "n_total": n_games}
    def _liveness_loop():
        i = 0
        while _liveness_state["alive"]:
            try:
                print(f"[cis-w{worker_id} LIVE +{i*5}s iter={iter_n}] "
                      f"n_done={_liveness_state['n_done']}/{_liveness_state['n_total']}",
                      flush=True)
                heartbeat_fn(iter_n=iter_n,
                             n_done=_liveness_state["n_done"],
                             n_total=_liveness_state["n_total"])
            except Exception:
                pass
            i += 1
            time.sleep(5.0)
    _liveness_thread = threading.Thread(target=_liveness_loop, daemon=True)
    _liveness_thread.start()

    heartbeat_fn(iter_n=iter_n, n_done=0, n_total=n_games)
    t0 = time.time()
    trajs, w, l, ties, summary, wr_per_opp, n_fft_w, n_fft_l = _run_collect_in_worker_cis(
        cis_handle=state["cis_handle"],
        device=device,
        worker_id=worker_id,
        iter_n=iter_n,
        n_games=n_games,
        max_concurrent=max_concurrent,
        server_url=server_url,
        opp_pool=opp_pool,
        temp_range=temp_range,
        opp_temp_range=opp_temp_range,
        fp16=fp16,
        rs_cfg=rs_cfg,
        turn_cap=turn_cap,
        battle_format=battle_format,
        procedural_teams_path=procedural_teams_path,
        heartbeat_fn=heartbeat_fn,
        liveness_state=_liveness_state,
    )
    elapsed_s = time.time() - t0
    _liveness_state["alive"] = False

    # Write traj file atomically (same format as mp_disk).
    traj_path = f"/tmp/traj_cis_w{worker_id}_iter{iter_n}.pkl.gz"
    tmp_path = traj_path + ".tmp"
    bundle = {
        "trajectories": trajs, "iter_n": iter_n, "worker_id": worker_id,
        "n_games": w + l + ties, "wr_per_opp": wr_per_opp,
        "elapsed_s": elapsed_s,
    }
    with gzip.open(tmp_path, "wb", compresslevel=1) as f:
        _pickle.dump(bundle, f, protocol=_pickle.HIGHEST_PROTOCOL)
    _os.replace(tmp_path, traj_path)

    result_pipe.send({
        "status": "done", "worker_id": worker_id, "iter_n": iter_n,
        "traj_path": traj_path,
        "n_games_played": w + l + ties,
        "wins": w, "losses": l, "ties": ties,
        "n_forfeit_wins": n_fft_w, "n_forfeit_losses": n_fft_l,
        "wr_per_opp": wr_per_opp, "elapsed_s": elapsed_s,
    })


def _run_collect_in_worker_cis(*, cis_handle, device, worker_id, iter_n,
                               n_games, max_concurrent, server_url, opp_pool,
                               temp_range, opp_temp_range, fp16, rs_cfg,
                               turn_cap, battle_format, procedural_teams_path,
                               heartbeat_fn, liveness_state=None):
    """CIS-routed self-play collect inside one worker.

    Differences from mp_disk's _run_collect_in_worker:
    - No model load (CIS owns the model)
    - No opp ckpt cache (CIS loads opp via reload_weights when self-play
      switches opp - or, in Phase 4.1, just self-play vs SAME model and
      defer opp routing to Phase 4.2)
    - Uses CISInferenceBatcher instead of InferenceBatcher

    Memory hygiene fixes (3.5b/c) + POKE_LOOP threading rule (await
    asyncio.sleep(1.5)) all carry over from mp_disk.
    """
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from rl_player import V9RLPlayer, make_self_play_opponent
    from rl_collection import _make_server
    from teams_ou import random_pool_teambuilder
    from team_generator import procedural_teambuilder
    from ppo import _cancel_listener
    import asyncio
    import gc

    srv = _make_server(server_url)

    # PFSP weights from main.
    weights_arr = [max(o.get("weight", 1.0), 1e-6) for o in opp_pool]
    total_w = sum(weights_arr)
    fractions = [w_ / total_w for w_ in weights_arr]
    n_per_opp = [max(1, int(round(n_games * fr))) for fr in fractions]
    diff = n_games - sum(n_per_opp)
    if diff != 0:
        n_per_opp[0] += diff

    if procedural_teams_path:
        train_tb = procedural_teambuilder(procedural_teams_path)
    else:
        train_tb = None

    # Build CIS-routed inference batcher.
    batcher = CISInferenceBatcher(
        handle=cis_handle, device=device, fp16=fp16,
        min_batch=min(8, max_concurrent), timeout_ms=15,
    )

    all_trajs = []
    total_w_count = total_l = total_ties = 0
    total_fft_w = total_fft_l = 0
    wr_per_opp: Dict[str, Dict[str, int]] = {}
    n_done = 0

    async def _play_vs_opp(opp_entry: dict, n_for_opp: int):
        nonlocal n_done, total_w_count, total_l, total_ties, total_fft_w, total_fft_l
        opp_path = opp_entry["path"]
        batch_id = (worker_id * 100000) + (iter_n * 1000) + (hash(opp_path) % 1000)
        opp_tb = train_tb or random_pool_teambuilder()

        player = V9RLPlayer(
            batcher=batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=turn_cap,
            battle_format=battle_format,
            team=train_tb or random_pool_teambuilder(),
            max_concurrent_battles=min(max_concurrent, n_for_opp),
            account_configuration=AccountConfiguration(
                f"CISw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        # Phase 4.1 simplification: opp is the SAME CISInferenceBatcher (i.e.,
        # self-play vs SAME current model, not vs an opp ckpt). PFSP weighting
        # is approximated by replaying current vs current. Phase 4.2 will add
        # opp ckpt swap via CIS handle to a separate "opp model" slot.
        # Practical impact: lose true PFSP-against-old-snapshots in CIS path
        # until Phase 4.2 ships. Production keeps using mp_disk path until then.
        opponent = V9RLPlayer(
            batcher=batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=turn_cap,
            battle_format=battle_format,
            team=opp_tb,
            max_concurrent_battles=min(max_concurrent, n_for_opp),
            account_configuration=AccountConfiguration(
                f"CISo{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=n_for_opp),
                timeout=max(300, n_for_opp * 30),
            )
        except asyncio.TimeoutError:
            print(f"[cis-w{worker_id}] timeout vs {opp_path}", flush=True)
        except Exception as e:
            print(f"[cis-w{worker_id}] error vs {opp_path}: {e}", flush=True)

        opp_w = sum(1 for b in player.battles.values() if b.won is True
                    and player._finish_looks_real(b))
        opp_l = sum(1 for b in player.battles.values() if b.won is False
                    and player._finish_looks_real(b))
        opp_t = sum(1 for b in player.battles.values() if b.won is None)
        opp_fft_w = getattr(player, 'n_forfeit_wins', 0)
        opp_fft_l = getattr(player, 'n_forfeit_losses', 0)

        all_trajs.extend(player.completed_trajectories)
        total_w_count += opp_w
        total_l += opp_l
        total_ties += opp_t
        total_fft_w += opp_fft_w
        total_fft_l += opp_fft_l
        wr_per_opp[opp_path] = {
            "w": opp_w, "g": opp_w + opp_l,
            "fft_w": opp_fft_w, "fft_l": opp_fft_l,
        }
        n_done += n_for_opp
        if liveness_state is not None:
            liveness_state["n_done"] = n_done

        try:
            player.reset_battles()
            opponent.reset_battles()
        except EnvironmentError:
            pass
        try:
            _cancel_listener(player)
            _cancel_listener(opponent)
        except Exception:
            pass
        del player, opponent

        # POKE_LOOP threading rule (cookbook §3j): yield wall-clock time
        # so POKE_LOOP can drain cancellations + close websockets. Skip
        # this and the worker hangs at 99% CPU. SAME RULE as mp_disk.
        await asyncio.sleep(1.5)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    async def _heartbeat_during_collect():
        loop_counter = 0
        while True:
            heartbeat_fn(iter_n=iter_n, n_done=n_done, n_total=n_games)
            try:
                if hasattr(batcher, '_prof_total_requests'):
                    print(f"[cis-w{worker_id} t+{loop_counter*15}s] "
                          f"infer_reqs={batcher._prof_total_requests} "
                          f"n_done={n_done}/{n_games}", flush=True)
            except Exception:
                pass
            loop_counter += 1
            await asyncio.sleep(15.0)

    async def _main():
        hb_task = asyncio.create_task(_heartbeat_during_collect())
        try:
            opp_coros = []
            for opp_entry, n_for in zip(opp_pool, n_per_opp):
                if n_for <= 0:
                    continue
                opp_coros.append(_play_vs_opp(opp_entry, n_for))
            print(f"[cis-w{worker_id}] gather start: {len(opp_coros)} opp coros, "
                  f"n_per_opp={n_per_opp}", flush=True)
            results = await asyncio.gather(*opp_coros, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    import traceback as _tb
                    print(f"[cis-w{worker_id}] opp {i} EXCEPTION: "
                          f"{type(r).__name__}: {r}\n"
                          f"{''.join(_tb.format_exception(type(r), r, r.__traceback__))}",
                          flush=True)
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()

    summary = {"worker_id": worker_id, "iter_n": iter_n}
    return (all_trajs, total_w_count, total_l, total_ties, summary,
            wr_per_opp, total_fft_w, total_fft_l)


# =============================
# Phase 4: orchestrator - mp_centralized_collect_sync
# =============================
#
# Drop-in replacement for mp_disk_collect_sync when --cis flag is set.
# Key differences from mp_disk:
#   1. Spawn ONE CIS subprocess (owns the GPU model) at first call
#   2. Spawn N worker subprocesses, each holding ONE CISClientHandle
#   3. Workers do inference via CIS, not their own GPU model
#   4. Weight sync: main writes weights to disk, signals CIS reload (one
#      reload per iter; cheaper than N worker reloads in mp_disk)
#
# Lifecycle: CIS + workers persist across iters as a singleton, same as
# mp_disk's _GLOBAL_MANAGER pattern.

_CIS_GLOBAL: Optional[Dict[str, Any]] = None  # holds CISServer, worker procs, pipes


def shutdown_cis_workers() -> None:
    """End-of-run cleanup. Safe to call repeatedly."""
    global _CIS_GLOBAL
    if _CIS_GLOBAL is None:
        return
    g = _CIS_GLOBAL
    _CIS_GLOBAL = None

    # Shutdown workers first (they need pipes to CIS that we're about to close).
    for wid, proc in g.get("workers", {}).items():
        try:
            g["worker_ctrl_pipes"][wid].send({"cmd": "shutdown"})
        except Exception:
            pass
    time.sleep(2.0)
    for wid, proc in g.get("workers", {}).items():
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()

    # Then shutdown CIS server.
    if g.get("server") is not None:
        try:
            g["server"].shutdown()
        except Exception:
            pass

    # Close any leftover pipe parent ends.
    for d in [g.get("worker_ctrl_pipes", {}), g.get("worker_result_pipes", {})]:
        for p in d.values():
            try:
                p.close()
            except Exception:
                pass


def _ensure_cis_global(weights_path: str, n_workers: int, device: str,
                       fp16: bool, amp_dtype_name: Optional[str],
                       min_batch: int, timeout_ms: int) -> Dict[str, Any]:
    """Lazy-init the CIS server + N worker procs. Returns the global dict."""
    global _CIS_GLOBAL
    if _CIS_GLOBAL is not None:
        return _CIS_GLOBAL

    ctx = _get_mp_ctx()

    # 1. Spawn CIS server with N pipes (one per worker)
    cis_server = CISServer(
        ckpt_path=weights_path, n_workers=n_workers,
        device=device, fp16=fp16, amp_dtype_name=amp_dtype_name,
        min_batch=min_batch, timeout_ms=timeout_ms,
    )
    cis_handles = cis_server.spawn(ready_timeout_s=120.0)

    # 2. Spawn N worker procs, each gets ONE handle's pipe ends.
    # The handle's req_writer (parent->worker side) and resp_reader belong
    # to MAIN. The worker needs the OTHER side of each pipe... wait, no.
    # The handle was created so that MAIN talks to CIS via these. For a
    # WORKER process to talk to CIS, it needs ITS OWN pair of pipes that
    # connect to CIS. CISServer.spawn already created N pairs - one per
    # worker. handles[i] gives main access to pipe pair i. But the WORKER
    # process should OWN those pipe ends instead of main.
    #
    # The pattern: main creates pipes, hands them to BOTH the CIS process
    # AND the worker. Worker keeps the parent ends; CIS got the child ends.
    # In _ensure_cis_global, we currently have main holding the parent
    # ends (via cis_handles[i].req_writer/resp_reader). To migrate, we'd
    # need to pickle the pipe ends and pass them to spawn'd workers.
    #
    # mp.Pipe ends ARE picklable (they're just file descriptors). So we
    # can pass them as args to the worker proc.

    workers: Dict[int, Any] = {}
    worker_ctrl_writers: Dict[int, Any] = {}  # main -> worker ctrl direction
    worker_result_readers: Dict[int, Any] = {}  # worker -> main result direction

    for wid in range(n_workers):
        # Worker's main-talks-to-it ctrl pipe
        ctrl_r, ctrl_w = ctx.Pipe(duplex=False)
        # Worker's it-talks-to-main result pipe
        res_r, res_w = ctx.Pipe(duplex=False)

        # The worker also gets THE CIS-bound pipes from the handle.
        # These were originally main-side; transfer them to the worker.
        cis_handle_for_worker = cis_handles[wid]
        cis_req_writer = cis_handle_for_worker.req_writer
        cis_resp_reader = cis_handle_for_worker.resp_reader

        proc = ctx.Process(
            target=_cis_worker_main,
            args=(wid, ctrl_r, res_w, cis_req_writer, cis_resp_reader),
            daemon=False,
        )
        proc.start()
        # Close child-side ends in parent
        ctrl_r.close()
        res_w.close()
        # The CIS pipe ends are now held by the worker (after fork/spawn
        # they're inherited); main can drop its references. But CIS handle
        # still has them in cis_server's _handles. We don't close them
        # here because the worker uses them.

        workers[wid] = proc
        worker_ctrl_writers[wid] = ctrl_w
        worker_result_readers[wid] = res_r

        # Pace the spawns (same SemLock-race mitigation pattern)
        time.sleep(0.5)

    time.sleep(2.0)

    _CIS_GLOBAL = {
        "server": cis_server,
        "handles_main_view": cis_handles,  # reference for weight reload
        "workers": workers,
        "worker_ctrl_pipes": worker_ctrl_writers,
        "worker_result_pipes": worker_result_readers,
        "n_workers": n_workers,
    }
    return _CIS_GLOBAL


def mp_centralized_collect_sync(
    model, device, server_pool, *,
    n_games: int,
    max_concurrent: int,
    snapshot_pool: List[str],
    fp16: bool,
    reward_shaper_cfg: dict,
    temp_range: Tuple[float, float],
    opponent_device: str,
    win_rates: Optional[dict],
    turn_cap: int,
    battle_format: str,
    procedural_teams_path: Optional[str],
    iter_n: int,
    n_workers: int = 8,
    rng_seed: Optional[int] = None,
    amp_dtype: Optional[str] = None,
    cis_min_batch: int = 8,
    cis_timeout_ms: int = 15,
) -> Tuple[List, int, int, int, Dict, float, dict]:
    """Synchronous one-iter CIS-routed collect.

    Drop-in replacement for mp_disk_collect_sync when --cis flag is set.
    Returns the same tuple shape: (trajs, w, l, t, steps, opp_name,
    elapsed, opp_records).
    """
    if device.type == "cpu":
        raise ValueError("--cis not supported on CPU; use --pipeline or sync collect.")
    if rng_seed is None:
        rng_seed = random.randint(0, 1_000_000)

    # 1. Save weights for CIS to load (reuse mp_disk's atomic write helper)
    weights_path = _save_weights_atomic_for_cis(model, iter_n)

    # 2. Init or reuse CIS global (spawn + workers on first iter)
    g = _ensure_cis_global(
        weights_path=weights_path, n_workers=n_workers,
        device=str(device), fp16=fp16, amp_dtype_name=amp_dtype,
        min_batch=cis_min_batch, timeout_ms=cis_timeout_ms,
    )

    # 3. Signal CIS to reload weights for this iter (skip on first call;
    # CIS already loaded the iter's weights at spawn).
    if iter_n > 0:  # rough heuristic; first call after iter 0 reloads
        try:
            g["handles_main_view"][0].reload(weights_path, timeout_s=60.0)
        except Exception as e:
            print(f"[cis-collect] WARN reload at iter {iter_n}: {e}", flush=True)

    # 4. Build opp_pool message + dispatch to workers
    opp_pool_msg = _build_opp_pool_msg(snapshot_pool, win_rates)

    n_alive = n_workers  # CIS workers don't have respawn yet (Phase 4.2 work)
    games_per_worker = n_games // n_alive
    remainder = n_games % n_alive

    cmds_sent: List[int] = []
    t_start = time.time()
    for i, wid in enumerate(range(n_workers)):
        worker_n = games_per_worker + (1 if i < remainder else 0)
        srv_idx = i % len(server_pool)
        srv = server_pool[srv_idx]
        srv_url = getattr(srv, 'websocket_url', None) or str(srv)
        cmd = {
            "cmd": "collect_iter",
            "iter_n": iter_n,
            "n_games": worker_n,
            "max_concurrent": max_concurrent,
            "server_url": srv_url,
            "opp_pool": opp_pool_msg,
            "temp_range": list(temp_range),
            "opp_temp_range": list(temp_range),
            "fp16": fp16,
            "rs_cfg": reward_shaper_cfg,
            "turn_cap": turn_cap,
            "battle_format": battle_format,
            "procedural_teams_path": procedural_teams_path,
            "device": str(device),
            "opponent_device": opponent_device,
            "rng_seed": rng_seed,
            "amp_dtype": amp_dtype,
        }
        g["worker_ctrl_pipes"][wid].send(cmd)
        cmds_sent.append(wid)
        # Stagger same as mp_disk - reduces I/O burst at iter boundary.
        if i + 1 < n_workers:
            time.sleep(0.25)

    # 5. Watchdog loop: collect results
    from multiprocessing.connection import wait as mp_wait
    expected = len(cmds_sent)
    received = 0
    results: List[dict] = []
    expected_collect_s = max(300.0, n_games / max(n_alive, 1) * 2.0)
    deadline = time.time() + 4.0 * expected_collect_s

    pipes_to_wid = {g["worker_result_pipes"][wid]: wid for wid in cmds_sent}

    while received < expected and time.time() < deadline:
        ready = mp_wait(list(pipes_to_wid.keys()), timeout=10.0)
        if not ready:
            continue
        for conn in ready:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                continue
            status = msg.get("status")
            if status == "heartbeat":
                continue
            received += 1
            if status == "done":
                results.append(msg)
            elif status == "error":
                print(f"[cis-collect] iter {iter_n}: worker {msg.get('worker_id')} "
                      f"error: {msg.get('exc_msg')}\n{msg.get('traceback', '')}",
                      flush=True)

    # 6. Aggregate
    import gzip as _gzip
    import pickle as _pickle
    all_trajs = []
    total_w = total_l = total_ties = 0
    aggregated_wr: Dict[str, List[int]] = {}
    for r in results:
        traj_path = r.get("traj_path")
        if not traj_path or not Path(traj_path).exists():
            continue
        try:
            with _gzip.open(traj_path, "rb") as f:
                bundle = _pickle.load(f)
            all_trajs.extend(bundle["trajectories"])
            total_w += r["wins"]
            total_l += r["losses"]
            total_ties += r["ties"]
            for opp_path, stats in r.get("wr_per_opp", {}).items():
                rec = aggregated_wr.setdefault(opp_path, [0, 0])
                rec[0] += stats.get("w", 0)
                rec[1] += stats.get("g", 0)
        except Exception as e:
            print(f"[cis-collect] read failure for {traj_path}: {e}", flush=True)

    # Cleanup old files (keep last 2 iters)
    threshold = iter_n - 2
    for f in Path("/tmp").glob("traj_cis_w*_iter*.pkl.gz"):
        try:
            stem = f.name.replace(".pkl.gz", "")
            n = int(stem.split("_iter")[-1])
            if n < threshold:
                f.unlink()
        except (ValueError, OSError):
            pass

    elapsed = time.time() - t_start
    total_steps = sum(len(t) for t in all_trajs)
    opp_name = f"cis(N={n_workers},responded={len(results)})"
    return (all_trajs, total_w, total_l, total_ties, total_steps, opp_name,
            elapsed, aggregated_wr)


def _build_opp_pool_msg(snapshot_pool, win_rates) -> List[dict]:
    """Build PFSP-weighted opp_pool list for the cmd dict. Same shape as
    mp_disk_collect's helper of the same name."""
    out = []
    for path in snapshot_pool:
        wr_entry = (win_rates or {}).get(path, {})
        if isinstance(wr_entry, dict):
            w = wr_entry.get("w", 0)
            g = max(wr_entry.get("g", 0), 1)
            wr = w / g
        else:
            wr = 0.5
        weight = (1.0 - wr) ** 2
        out.append({"path": path, "wr": wr, "weight": weight})
    return out


def _save_weights_atomic_for_cis(model, iter_n: int) -> str:
    """Atomic weights write for CIS reload. Mirrors mp_disk's
    _save_worker_weights_atomic. Strips _orig_mod prefix on save (CIS's
    reload also strips, but doing it on write keeps disk format clean
    and matches the no-compile-prefix convention)."""
    final = f"/tmp/cis_weights_iter{iter_n}.pt"
    tmp = final + ".tmp"
    sd = model.state_dict()
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    cfg_dict = model.cfg.to_dict() if hasattr(model.cfg, "to_dict") else None
    torch.save({
        "model_state_dict": sd,
        "model_config": cfg_dict,
        "arch": "transformer",
    }, tmp)
    try:
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp, final)
    return final


# =============================
# Phase 1 self-test (run as script)
# =============================

if __name__ == "__main__":
    # Quick smoke: spawn CIS, ping, shutdown.
    # Full logits-identity test lives at scripts/diag/test_cis_phase1.py.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cis = CISClient(args.ckpt, device=args.device, fp16=True)
    cis.spawn(ready_timeout_s=120.0)
    print("CIS up")
    print("ping:", "OK" if cis.ping() else "FAIL")
    cis.shutdown()
    print("CIS shutdown OK")
