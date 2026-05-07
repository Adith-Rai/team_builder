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

    # Service loop with cross-worker batching
    pending: list = []  # list of (worker_idx, req_id_or_None, batch_size, np_batch)
    last_fire_t = time.time()
    timeout_s = timeout_ms / 1000.0
    closed_workers: set = set()

    def _fire_batch():
        """Fire one batched forward over `pending` and dispatch responses."""
        nonlocal last_fire_t
        if not pending:
            return
        try:
            stacked = _stack_numpy_batches([p[3] for p in pending])
            torch_batch = numpy_dict_to_torch(stacked, device)
            with torch.no_grad(), autocast_ctx(fp16):
                out = model(torch_batch)
            mega_np = _output_to_numpy(out)

            # Dispatch per-request slices
            cursor = 0
            for (widx, req_id, bsize, _) in pending:
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
            for (widx, req_id, _, _) in pending:
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
                pending.append((widx, req.get("req_id"), bsize, np_batch))

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
