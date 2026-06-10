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
import random
import sys
import time
import traceback
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
                    ckpt_paths: List[str], device_str: str,
                    fp16: bool = True,
                    amp_dtype_name: Optional[str] = None,
                    min_batch: int = 8,
                    timeout_ms: int = 15,
                    ctrl_req_reader=None,
                    ctrl_resp_writer=None) -> None:
    """Phase 2/4.3a CIS subprocess entrypoint. Multiplexes inference requests
    across N worker pipes via mp.connection.wait. Holds K+1 model slots:
    slot 0 = player, slots 1..K = PFSP opp pool entries (Phase 4.3a).

    Each request carries a `slot` field (default 0). Pending requests are
    queued PER SLOT and batched independently — slot 0 (player) cross-batches
    requests from all 8 workers; slots 1..K cross-batch the subset of workers
    currently playing each opp. Batch fires when slot accumulates >= min_batch
    OR timeout_ms elapses since the slot's last fire.

    Protocol per pipe:
      Inbound on req_pipe[i]:
        {"cmd": "infer", "slot": int, "batch": dict, "history"?, "history_lens"?, "req_id"?}
        {"cmd": "reload", "slot": int, "weights_path": str}
        {"cmd": "ping"}
        {"cmd": "shutdown"}
      Outbound on resp_pipe[i]: per-request response; batched infer slices
        the mega-batch result and sends per-worker slices.

    Args:
      worker_req_readers:  list of mp.Pipe reader ends, one per worker
      worker_resp_writers: list of mp.Pipe writer ends, one per worker
      ckpt_paths: list of K+1 ckpt paths. Length is the number of slots.
                  Single-element list is the Phase 1-3 single-slot mode.
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

    # Phase 4.5 (S54): separate control pipe between CIS and PARENT (CISServer).
    # Used for reload calls that previously raced with worker recv_loops on
    # shared resp pipes. ctrl_req_reader is the read end CIS uses for parent
    # commands; ctrl_resp_writer is the write end CIS uses for parent
    # responses. Worker procs never see these pipes — no race possible.
    # Backward-compat: if ctrl pipes not provided, fall back to old behavior.
    _CTRL_WIDX = -1  # sentinel: ctrl path, not a worker

    n_slots = len(ckpt_paths)
    if n_slots < 1:
        raise ValueError("CIS needs at least one ckpt path")

    try:
        model_slots: List[Any] = []
        cfg = None
        for s, path in enumerate(ckpt_paths):
            m, cfg_s, _ = load_checkpoint(path, device)
            m.eval()
            model_slots.append(m)
            if cfg is None:
                cfg = cfg_s
        n_params = sum(p.numel() for p in model_slots[0].parameters())
        print(f"[CIS-multi N={n_workers} slots={n_slots}] loaded "
              f"{n_slots} model slot(s) ({n_params/1e6:.1f}M params each, "
              f"device={device}, min_batch={min_batch}, "
              f"timeout_ms={timeout_ms})", flush=True)
    except Exception as e:
        for w in worker_resp_writers:
            try:
                w.send({"status": "fatal",
                        "exc_msg": f"CIS failed to load ckpt: {e}",
                        "traceback": traceback.format_exc()})
            except Exception:
                pass
        return

    # Phase 4.3a uses slot 0 as the canonical "model" reference for shape
    # introspection. All slots share architecture (same TransformerConfig),
    # only the parameter values differ.
    model = model_slots[0]

    # CUDA stream priority (Phase 4.2): CIS forwards run on a LOW-priority
    # stream so when main process does optimizer.step on the default
    # (HIGH-priority) stream, CUDA scheduler honors priority - main gets
    # GPU first, CIS fills gaps. This is the KEY mechanism that makes
    # --pipeline overlap work without GPU contention deadlock (cookbook
    # §3c). Without priority, both compete unmanaged.
    #
    # Negative priorities = lower priority (closer to 0 = higher). Default
    # stream is priority 0; -1 is below default. On RunPod A100 the actual
    # scheduler honor depends on driver/CUDA version - if priorities are
    # ignored, CIS still works (just doesn't get the contention benefit).
    cis_low_pri_stream: Optional["torch.cuda.Stream"] = None
    if device.type == "cuda":
        try:
            cis_low_pri_stream = torch.cuda.Stream(device=device, priority=-1)
            print(f"[CIS-multi] low-priority CUDA stream created (priority=-1)",
                  flush=True)
        except Exception as e:
            print(f"[CIS-multi] WARN low-priority stream creation failed: {e} - "
                  f"using default stream (no scheduler arbitration)", flush=True)
            cis_low_pri_stream = None

    # Send ready to each worker's resp pipe so each handle gets a clean signal.
    for w in worker_resp_writers:
        try:
            w.send({"status": "ready"})
        except Exception:
            pass
    # Phase 4.5: also send ready on the ctrl pipe (if present) so CISServer's
    # ctrl handle can confirm liveness.
    if ctrl_resp_writer is not None:
        try:
            ctrl_resp_writer.send({"status": "ready"})
        except Exception:
            pass

    # Service loop with per-slot cross-worker batching.
    # pending_per_slot[s] is a list of 5-tuples
    # (worker_idx, req_id_or_None, batch_size, np_batch, np_history_or_None)
    # for slot s. Each slot accumulates and fires independently.
    pending_per_slot: List[list] = [[] for _ in range(n_slots)]
    last_fire_t_per_slot: List[float] = [time.time()] * n_slots
    timeout_s = timeout_ms / 1000.0
    closed_workers: set = set()

    # S58 CIS latency instrumentation — proves/disproves the "low-pri stream
    # contention under multi-slot load" hypothesis. Tracks per-slot fire
    # latency + queue depth, dumps aggregate stats every 60s. Zero behavioral
    # impact; pure timing/counters around the existing fire path.
    # S62: also tracks fire-trigger type (timeout vs min_batch) + mean queue
    # size — H3 (slot starvation) evidence for the Fix #1 design decision.
    _cis_stats: List[Dict[str, float]] = [
        {"n_fires": 0, "total_ms": 0.0, "max_ms": 0.0, "max_q": 0,
         "fwd_total_ms": 0.0, "n_timeout_fires": 0, "n_minbatch_fires": 0,
         "q_total": 0}
        for _ in range(n_slots)
    ]
    _last_stats_dump_t: List[float] = [time.time()]  # mutable single-element box

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

    def _maybe_dump_cis_stats():
        """S58: emit per-slot fire latency stats once every 60s. Resets
        counters after each dump so each window stands alone.
        S62: also emits fire-type breakdown (timeout vs min_batch) +
        mean queue size — slot-starvation diagnostic for Fix #1."""
        now = time.time()
        if now - _last_stats_dump_t[0] < 60.0:
            return
        _last_stats_dump_t[0] = now
        for s in range(n_slots):
            st = _cis_stats[s]
            if st["n_fires"] == 0:
                continue
            mean = st["total_ms"] / st["n_fires"]
            fwd_mean = st["fwd_total_ms"] / st["n_fires"]
            mean_q = st["q_total"] / st["n_fires"]
            n_to = int(st["n_timeout_fires"])
            n_mb = int(st["n_minbatch_fires"])
            to_pct = 100.0 * n_to / max(1, n_to + n_mb)
            print(f"[CIS-STATS] slot={s} fires={int(st['n_fires'])} "
                  f"mean_fire={mean:.1f}ms (fwd={fwd_mean:.1f}ms) "
                  f"max_fire={st['max_ms']:.1f}ms maxq={int(st['max_q'])} "
                  f"meanq={mean_q:.1f} timeout_pct={to_pct:.0f}% "
                  f"(to={n_to}/mb={n_mb})",
                  flush=True)
            st["n_fires"] = 0
            st["total_ms"] = 0.0
            st["fwd_total_ms"] = 0.0
            st["max_ms"] = 0.0
            st["max_q"] = 0
            st["n_timeout_fires"] = 0
            st["n_minbatch_fires"] = 0
            st["q_total"] = 0

    def _fire_batch(slot: int, fire_reason: str = "unknown"):
        """Fire one batched forward over `pending_per_slot[slot]` using
        `model_slots[slot]` and dispatch responses. Slot-isolated: requests
        for different slots batch independently.

        fire_reason (S62): "minbatch" if min_batch threshold hit, "timeout"
        if timeout fired with sub-threshold queue. Used for slot-starvation
        diagnostic in CIS-STATS dump.

        Two paths (per slot):
        - If NO request in this slot's pending has a history, use the simple
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
        pending = pending_per_slot[slot]
        slot_model = model_slots[slot]
        if not pending:
            return
        # S58 instrumentation: capture queue depth + start times for stats
        _n_reqs_at_fire = len(pending)
        _t_fire_start = time.perf_counter()
        _t_fwd_start = None
        _t_fwd_end = None
        try:
            # Are any requests carrying a non-empty history?
            any_history = any(p[4] is not None and p[4].shape[1] > 0
                              for p in pending)

            stacked = _stack_numpy_batches([p[3] for p in pending])
            torch_batch = numpy_dict_to_torch(stacked, device)
            _t_fwd_start = time.perf_counter()

            # Phase 4.2: run forward on LOW-priority CUDA stream so main
            # process's optimizer.step gets first dibs on the GPU. The
            # `with torch.cuda.stream(...)` ctx is a no-op if stream is None
            # (CPU device or stream creation failed at startup).
            # Phase 4.3a: stream is shared across slots (single low-priority
            # stream serves all slot forwards; slot just selects which model
            # to run, not which CUDA stream).
            _stream_ctx = (torch.cuda.stream(cis_low_pri_stream)
                           if cis_low_pri_stream is not None
                           else _nullcontext())
            with torch.no_grad(), autocast_ctx(fp16), _stream_ctx:
                if not any_history or not _have_arch_compat:
                    # Simple path: history-free or arch_compat unavailable.
                    out = slot_model(torch_batch)
                    mega_np = _output_to_numpy(out)
                else:
                    # Full batched-with-histories path. Mirrors
                    # inference_batcher._gpu_forward.
                    N = len(pending)
                    spatial_out, summaries = slot_model.forward_spatial(torch_batch)
                    action_ctx = call_action_encoder(slot_model, torch_batch, spatial_out)

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
                    max_T = min(max(seq_lens), slot_model.temporal.temporal_context)
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

                    temporal_ctx = slot_model.temporal(
                        all_summaries.float(), seq_lens_t
                    ).to(summaries.dtype)

                    actor_out = spatial_out[:, 0, :]
                    at = torch.cat([actor_out, temporal_ctx], dim=-1)
                    at_exp = at.unsqueeze(1).expand(-1, 9, -1)
                    pi_input = torch.cat([at_exp, action_ctx], dim=-1)
                    logits = call_policy_logits(slot_model, pi_input)

                    if "legal_mask" in torch_batch:
                        logits = logits.float().masked_fill(
                            torch_batch["legal_mask"] < 0.5, -100.0)

                    critic_out = spatial_out[:, 1, :]
                    vi = torch.cat([critic_out, temporal_ctx], dim=-1)
                    v_logits = call_value_logits(slot_model, vi)
                    v_probs = F.softmax(v_logits, dim=-1)
                    values = (v_probs * get_v_support(slot_model)).sum(-1)

                    mega_np = {
                        "action_logits": logits.detach().float().cpu().numpy(),
                        "value":         values.detach().float().cpu().numpy(),
                        "v_logits":      v_logits.detach().float().cpu().numpy(),
                        "summary":       summaries.detach().float().cpu().numpy(),
                    }

            # S58: GPU work has synchronized via .cpu().numpy() calls inside
            # the with block; mark the forward end time before CPU dispatch.
            _t_fwd_end = time.perf_counter()

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

        # S58 instrumentation: update per-slot latency stats
        # S62: also track fire-trigger type + cumulative queue for mean_q
        _t_fire_end = time.perf_counter()
        _fire_ms = (_t_fire_end - _t_fire_start) * 1000.0
        _fwd_ms = (((_t_fwd_end or _t_fire_end) -
                    (_t_fwd_start or _t_fire_start)) * 1000.0)
        st = _cis_stats[slot]
        st["n_fires"] += 1
        st["total_ms"] += _fire_ms
        st["fwd_total_ms"] += _fwd_ms
        st["q_total"] += _n_reqs_at_fire
        if fire_reason == "timeout":
            st["n_timeout_fires"] += 1
        elif fire_reason == "minbatch":
            st["n_minbatch_fires"] += 1
        if _fire_ms > st["max_ms"]:
            st["max_ms"] = _fire_ms
        if _n_reqs_at_fire > st["max_q"]:
            st["max_q"] = _n_reqs_at_fire

        pending_per_slot[slot].clear()
        last_fire_t_per_slot[slot] = time.time()

    while True:
        # Active pipes = worker readers + ctrl pipe (if present).
        active_pipes = [p for p in worker_req_readers
                        if pipe_to_widx[p] not in closed_workers]
        if ctrl_req_reader is not None:
            active_pipes.append(ctrl_req_reader)
        if not active_pipes:
            print(f"[CIS-multi] all worker pipes closed, exiting", flush=True)
            break

        # Compute earliest deadline across all slots with pending. We must
        # wake up by then so a pending slot's timeout-fire isn't starved
        # by quiet pipes elsewhere.
        now = time.time()
        any_pending = False
        earliest_deadline = now + timeout_s
        for s in range(n_slots):
            if pending_per_slot[s]:
                any_pending = True
                slot_deadline = last_fire_t_per_slot[s] + timeout_s
                if slot_deadline < earliest_deadline:
                    earliest_deadline = slot_deadline
        if any_pending:
            wait_to = max(0.0, earliest_deadline - now)
        else:
            wait_to = timeout_s

        ready = mp_wait(active_pipes, timeout=wait_to)

        for r in ready:
            # Determine if this is the ctrl pipe or a worker pipe.
            is_ctrl = (r is ctrl_req_reader)
            widx = _CTRL_WIDX if is_ctrl else pipe_to_widx[r]
            try:
                req = r.recv()
            except (EOFError, BrokenPipeError):
                if is_ctrl:
                    # Parent closed ctrl pipe (shouldn't happen until shutdown).
                    print(f"[CIS-multi] ctrl pipe closed unexpectedly", flush=True)
                    ctrl_req_reader = None  # stop polling it
                else:
                    closed_workers.add(widx)
                continue

            cmd = req.get("cmd") if isinstance(req, dict) else None

            # Helper: send response on the correct pipe (ctrl or worker).
            def _send_resp(msg):
                try:
                    if is_ctrl:
                        if ctrl_resp_writer is not None:
                            ctrl_resp_writer.send(msg)
                    else:
                        worker_resp_writers[widx].send(msg)
                except Exception:
                    if not is_ctrl:
                        closed_workers.add(widx)

            if cmd == "shutdown":
                if not is_ctrl:
                    closed_workers.add(widx)
                continue

            if cmd == "ping":
                resp = {"status": "ok"}
                if req.get("req_id") is not None:
                    resp["req_id"] = req.get("req_id")
                _send_resp(resp)
                continue

            if cmd == "reload":
                # Phase 3: hot-reload weights from disk. Main writes weights
                # atomically to a path (same pattern as mp_disk_collect_sync's
                # _save_worker_weights_atomic), sends "reload" with the path,
                # and CIS does load_state_dict in-place. The model architecture
                # is unchanged - only the parameters update. Returns "ok" once
                # load completes so main knows it's safe to start the next
                # iter's collect.
                # Phase 4.3a: req carries `slot` (default 0 = player). Reload
                # affects only that slot; other slots remain bit-stable.
                try:
                    weights_path = req.get("weights_path")
                    slot = int(req.get("slot", 0))
                    if not weights_path:
                        raise ValueError("reload cmd missing weights_path")
                    if slot < 0 or slot >= n_slots:
                        raise ValueError(f"reload slot {slot} out of range "
                                         f"[0, {n_slots})")
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
                        model_slots[slot].load_state_dict(sd, strict=True)
                    except RuntimeError as load_err:
                        # If strict load fails, fall back to non-strict but
                        # surface the error so caller knows reload was partial.
                        result = model_slots[slot].load_state_dict(sd, strict=False)
                        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
                        print(f"[CIS] WARN reload slot {slot} had key mismatches: {load_err}",
                              flush=True)
                    del state, sd
                    import gc
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    msg = {"status": "ok",
                           "slot": slot,
                           "missing_keys": missing[:5],
                           "unexpected_keys": unexpected[:5]}
                    if req.get("req_id") is not None:
                        msg["req_id"] = req.get("req_id")
                    _send_resp(msg)
                except Exception as e:
                    tb = traceback.format_exc()
                    err_msg = {
                        "status": "error",
                        "exc_msg": f"reload failed: {e}",
                        "traceback": tb,
                    }
                    if req.get("req_id") is not None:
                        err_msg["req_id"] = req.get("req_id")
                    _send_resp(err_msg)
                continue

            if cmd == "infer":
                if is_ctrl:
                    # Infer should never come via ctrl pipe — that's parent-only,
                    # and parent doesn't run inference. Respond with error.
                    err_msg = {"status": "error",
                               "exc_msg": "infer cmd not allowed on ctrl pipe"}
                    if req.get("req_id") is not None:
                        err_msg["req_id"] = req.get("req_id")
                    _send_resp(err_msg)
                    continue
                np_batch = req["batch"]
                slot = int(req.get("slot", 0))
                if slot < 0 or slot >= n_slots:
                    # Out-of-range slot: respond with error so worker fails
                    # loudly rather than silently routing to slot 0.
                    err_msg = {"status": "error",
                               "exc_msg": f"infer slot {slot} out of "
                                          f"range [0, {n_slots})"}
                    if req.get("req_id") is not None:
                        err_msg["req_id"] = req.get("req_id")
                    _send_resp(err_msg)
                    continue
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
                pending_per_slot[slot].append(
                    (widx, req.get("req_id"), bsize, np_batch, np_history))

        # Fire any slot that hit threshold or timed out. Each slot fires
        # independently — slot 0 (player) typically fires first because it
        # accumulates fastest (all workers contribute), slots 1..K fire when
        # their subset of workers reaches min_batch or hits timeout.
        now = time.time()
        for s in range(n_slots):
            if not pending_per_slot[s]:
                continue
            accumulated = sum(p[2] for p in pending_per_slot[s])
            timed_out = (now - last_fire_t_per_slot[s]) >= timeout_s
            hit_minbatch = accumulated >= min_batch
            if hit_minbatch or timed_out:
                # S62: classify fire reason for slot-starvation diagnostic.
                # If both true, prefer "minbatch" — the threshold was the
                # reason CIS would have fired even without the timeout.
                reason = "minbatch" if hit_minbatch else "timeout"
                _fire_batch(s, reason)
        # S58: periodic CIS latency stats dump (every 60s). Reveals whether
        # multi-slot load is driving per-slot fire latency up vs holding it
        # constant — the proof for/against the low-pri stream contention
        # hypothesis. No-op outside the dump window.
        _maybe_dump_cis_stats()


class CISClientHandle:
    """Per-worker view of a CIS server. Holds one (req_writer, resp_reader)
    pipe pair that routes through a single shared CIS subprocess. Multiple
    CISClientHandles can call .infer() concurrently from different threads
    or processes; CIS multiplexes via mp.connection.wait.

    Phase 4.4 (S54): async-with-req_id-dispatch. Each request gets a
    monotonically-assigned req_id. Multiple threads send concurrently
    behind a SEND-only lock (~us-scale critical section). A single
    recv-loop thread per handle reads responses off the pipe and
    demultiplexes them to per-req_id futures. Each caller awaits its
    own future. Removes the per-handle Lock that Phase 4.3 introduced
    — that lock was correct but capped throughput at 1-in-flight per
    handle (~50% GPU utilization at production scale, S53 measured).

    Concurrency model:
    - Multiple threads can call infer/reload/ping in parallel
    - send_lock is held only across pickle+send (~tens of microseconds)
    - recv-loop demuxes responses, sets per-req_id futures
    - Cross-handle (cross-worker) concurrency unchanged — each worker has
      its own pipe pair + own recv-loop"""

    def __init__(self, req_writer, resp_reader, worker_idx: int,
                 start_recv: bool = True):
        import threading as _threading
        import itertools as _itertools
        self.req_writer = req_writer
        self.resp_reader = resp_reader
        self.worker_idx = worker_idx

        # Phase 4.4 async-dispatch state.
        self._send_lock = _threading.Lock()  # cheap atomic enqueue
        self._next_req_id = _itertools.count(1)  # 0 reserved for legacy/None
        self._pending: Dict[int, Any] = {}  # Dict[int, Future]
        self._pending_lock = _threading.Lock()
        self._stopped = False
        self._recv_thread: Optional[_threading.Thread] = None
        if start_recv:
            self.start_recv_loop()

    def start_recv_loop(self) -> None:
        """Start the response dispatcher thread. Called by CISServer.spawn()
        AFTER consuming the initial 'ready' message — otherwise the recv
        loop would steal it from spawn()'s direct resp_reader.recv() call.
        Idempotent: safe to call once; subsequent calls are no-ops."""
        import threading as _threading
        if self._recv_thread is not None:
            return
        self._recv_thread = _threading.Thread(
            target=self._recv_loop,
            name=f"CISClientHandle-recv-{self.worker_idx}",
            daemon=True,
        )
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        """Single thread per handle that polls resp_reader + dispatches
        responses to per-req_id futures. Exits when pipe closes (EOF)
        or shutdown() is called."""
        while not self._stopped:
            try:
                # Short poll so shutdown can take effect quickly.
                if not self.resp_reader.poll(timeout=0.1):
                    continue
                resp = self.resp_reader.recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                # Pipe closed. Fail all pending and exit.
                self._fail_all_pending(
                    RuntimeError(f"CIS pipe closed (worker {self.worker_idx}): "
                                 f"{type(e).__name__}")
                )
                return
            except Exception as e:
                # Unexpected error; fail all pending and exit.
                self._fail_all_pending(e)
                return

            req_id = resp.get("req_id") if isinstance(resp, dict) else None
            if req_id is None:
                # Untagged response (e.g., legacy callers). Drop with warn.
                # All Phase 4.4+ requests carry req_id, so this is rare.
                print(f"[CIS-handle {self.worker_idx}] WARN untagged response "
                      f"dropped: {resp!r}", flush=True)
                continue

            with self._pending_lock:
                fut = self._pending.pop(req_id, None)
            if fut is None:
                # Future was likely cleaned up due to caller timeout.
                # Discard the response.
                continue
            if not fut.done():
                fut.set_result(resp)

    def _fail_all_pending(self, exc: BaseException) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(exc)

    def _send_with_future(self, msg: Dict[str, Any]) -> "concurrent.futures.Future":
        """Assign req_id, register future, atomically send msg. Returns
        the future the caller should await."""
        from concurrent.futures import Future as _Future
        req_id = next(self._next_req_id)
        msg["req_id"] = req_id
        fut: "_Future" = _Future()
        with self._pending_lock:
            self._pending[req_id] = fut
        try:
            with self._send_lock:
                self.req_writer.send(msg)
        except Exception:
            # Send failed; remove future + propagate.
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise
        return fut

    def _await_future(self, fut, timeout_s: float, op: str) -> Dict[str, Any]:
        """Block until future resolves. Drop pending entry on timeout
        so a late response doesn't leak memory."""
        from concurrent.futures import TimeoutError as _FutTimeout
        try:
            return fut.result(timeout=timeout_s)
        except _FutTimeout:
            # Find the req_id and drop it from _pending.
            with self._pending_lock:
                stale_ids = [rid for rid, f in self._pending.items() if f is fut]
                for rid in stale_ids:
                    self._pending.pop(rid, None)
            raise TimeoutError(f"CIS {op} (worker {self.worker_idx}) "
                               f"response not received within {timeout_s}s")

    def _sync_send_recv(self, msg: Dict[str, Any], timeout_s: float,
                         op: str) -> Dict[str, Any]:
        """Lock-based send+recv for handles that did NOT start_recv_loop().
        Used by parent-process handles in CISServer (rare reload calls
        between iters; no concurrency on the parent side). Worker-side
        handles always go through the async-dispatch path."""
        with self._send_lock:
            self.req_writer.send(msg)
            if not self.resp_reader.poll(timeout=timeout_s):
                raise TimeoutError(f"CIS {op} (worker {self.worker_idx}) "
                                   f"response not received within {timeout_s}s")
            resp = self.resp_reader.recv()
        return resp

    def infer(self, numpy_batch: Dict[str, Any], timeout_s: float = 30.0,
              req_id: Optional[int] = None,
              history: Optional[np.ndarray] = None,
              history_lens: Optional[np.ndarray] = None,
              slot: int = 0) -> Dict[str, np.ndarray]:
        # `req_id` kwarg accepted for back-compat with Phase 4.3 callers
        # (tests that passed explicit req_ids); ignored — Phase 4.4
        # assigns its own req_id internally for the dispatcher to demux.
        msg: Dict[str, Any] = {"cmd": "infer", "batch": numpy_batch,
                                "slot": int(slot)}
        if history is not None:
            msg["history"] = history
        if history_lens is not None:
            msg["history_lens"] = history_lens
        if self._recv_thread is not None:
            # Async-dispatch path (worker-side handles).
            fut = self._send_with_future(msg)
            resp = self._await_future(fut, timeout_s, "infer")
        else:
            # Sync lock-based path (parent-side handles in CISServer; no
            # recv-thread because parent does its own resp_reader.recv()
            # for the initial 'ready' signal + occasional reload calls).
            resp = self._sync_send_recv(msg, timeout_s, "infer")
        status = resp.get("status")
        if status == "ok":
            return resp["out"]
        if status == "error":
            raise RuntimeError(f"CIS infer error (worker {self.worker_idx}): "
                               f"{resp.get('exc_msg')}\n{resp.get('traceback','')}")
        raise RuntimeError(f"CIS unexpected response: {resp!r}")

    def ping(self, timeout_s: float = 5.0) -> bool:
        try:
            if self._recv_thread is not None:
                fut = self._send_with_future({"cmd": "ping"})
                resp = self._await_future(fut, timeout_s, "ping")
            else:
                resp = self._sync_send_recv({"cmd": "ping"}, timeout_s, "ping")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def reload(self, weights_path: str, timeout_s: float = 60.0,
               slot: int = 0) -> Dict[str, Any]:
        """Phase 3: signal CIS to load fresh weights from disk. Blocks until
        CIS confirms load complete. Returns the response dict (with optional
        missing_keys/unexpected_keys for diagnostics).

        slot selects which model slot to reload (default 0 = player slot).
        Phase 4.3a adds K opp slots; orchestrator reloads slot 1..K with
        PFSP pool ckpts at iter start. Phase 4.4 routes through async
        dispatch (worker-side) or sync lock (parent-side)."""
        msg = {"cmd": "reload", "weights_path": weights_path, "slot": int(slot)}
        if self._recv_thread is not None:
            fut = self._send_with_future(msg)
            resp = self._await_future(fut, timeout_s, "reload")
        else:
            resp = self._sync_send_recv(msg, timeout_s, "reload")
        if resp.get("status") == "ok":
            return resp
        raise RuntimeError(f"CIS reload error: {resp.get('exc_msg')}\n"
                           f"{resp.get('traceback','')}")

    def shutdown(self) -> None:
        # Mark stopped first so recv_loop exits cleanly when pipe closes.
        self._stopped = True
        try:
            with self._send_lock:
                self.req_writer.send({"cmd": "shutdown"})
        except Exception:
            pass
        # Fail any still-pending futures so callers don't hang.
        self._fail_all_pending(RuntimeError(
            f"CIS handle (worker {self.worker_idx}) shutting down"))


class CISWorkerManager:
    """Phase 4.6 (S54): lifecycle manager for CIS worker procs.

    Owns spawn/health_check/kill_all. Does NOT do per-worker respawn —
    the unified Option B recovery path is full CIS+workers reset on any
    detected failure, handled by `_orchestrator_full_reset` at module
    level. See `_cis_wait_results_and_aggregate` for detection.

    HEARTBEAT_TIMEOUT_S=600.0 matches mp_disk WorkerManager — same
    iter-17-class hang RCA applies (heartbeat starvation when blocking
    torch.load races the asyncio loop)."""

    HEARTBEAT_TIMEOUT_S = 600.0

    def __init__(self, n_workers: int, ctx):
        self.n = n_workers
        self.ctx = ctx
        self.workers: Dict[int, Any] = {}
        # Parent-side ends of worker<->parent ctrl/result pipes.
        self.ctrl_pipes: Dict[int, Any] = {}     # parent writes -> worker reads
        self.result_pipes: Dict[int, Any] = {}   # parent reads <- worker writes
        # Health tracking.
        self.last_heartbeat: Dict[int, float] = {}

    def _spawn_proc(self, worker_id: int, cis_req_writer, cis_resp_reader):
        """Spawn one worker proc: creates ctrl/result pipes (parent<->worker),
        spawns the proc with the supplied CIS pipe ends, closes child-side
        ends in parent. Populates manager dicts."""
        ctrl_r, ctrl_w = self.ctx.Pipe(duplex=False)
        res_r, res_w = self.ctx.Pipe(duplex=False)
        proc = self.ctx.Process(
            target=_cis_worker_main,
            args=(worker_id, ctrl_r, res_w, cis_req_writer, cis_resp_reader),
            daemon=False,
        )
        proc.start()
        # Close child-side ends in parent (held only by worker after spawn).
        ctrl_r.close()
        res_w.close()
        self.workers[worker_id] = proc
        self.ctrl_pipes[worker_id] = ctrl_w
        self.result_pipes[worker_id] = res_r
        self.last_heartbeat[worker_id] = time.time()

    def initial_spawn(self, worker_id: int, cis_req_writer, cis_resp_reader,
                       pace_s: float = 0.5):
        """First-time spawn during _ensure_cis_global. Uses the CIS pipe
        ends pre-allocated by CISServer.spawn() (parent's view of the
        per-worker pipe pairs). Pacing matches mp_disk SemLock-race
        mitigation."""
        self._spawn_proc(worker_id, cis_req_writer, cis_resp_reader)
        if pace_s > 0:
            time.sleep(pace_s)

    def health_check(self) -> List[Tuple[int, str]]:
        """Returns [(worker_id, reason), ...] for any unhealthy worker.
        Caller (orchestrator) treats any non-empty result as a trigger
        for full reset — no per-worker respawn."""
        unhealthy = []
        now = time.time()
        for wid, p in self.workers.items():
            if not p.is_alive():
                unhealthy.append((wid, "dead"))
            elif now - self.last_heartbeat.get(wid, now) > self.HEARTBEAT_TIMEOUT_S:
                unhealthy.append((wid, "stale_heartbeat"))
        return unhealthy

    def alive_worker_ids(self) -> List[int]:
        return [wid for wid in self.workers
                if self.workers[wid].is_alive()]

    def kill_all(self):
        """End-of-run cleanup OR pre-reset cleanup. Sends shutdown cmd,
        terminates, joins."""
        for wid, ctrl in list(self.ctrl_pipes.items()):
            try:
                ctrl.send({"cmd": "shutdown"})
            except Exception:
                pass
        time.sleep(2.0)
        for wid, p in list(self.workers.items()):
            if p.is_alive():
                p.terminate()
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
        for d in (self.ctrl_pipes, self.result_pipes):
            for p in d.values():
                try:
                    p.close()
                except Exception:
                    pass
        self.workers.clear()
        self.ctrl_pipes.clear()
        self.result_pipes.clear()


class CISServer:
    """Phase 2 server: spawns one CIS subprocess that multiplexes N worker pipes
    with cross-worker batching. Returns N CISClientHandle objects, one per
    worker, that can be used independently from threads/processes."""

    def __init__(self, ckpt_path: Union[str, List[str]], n_workers: int,
                 device: str = "cuda",
                 fp16: bool = True, amp_dtype_name: Optional[str] = None,
                 min_batch: int = 8, timeout_ms: int = 15):
        # Phase 4.3a: ckpt_path can be a single str (single-slot, back-compat
        # with Phase 1-3) or a list of str (multi-slot: slot 0 = player,
        # slots 1..K = PFSP opp pool entries). Single str is normalized to
        # a 1-element list internally so the server always sees List[str].
        if isinstance(ckpt_path, str):
            self.ckpt_paths: List[str] = [ckpt_path]
        else:
            self.ckpt_paths = list(ckpt_path)
            if not self.ckpt_paths:
                raise ValueError("CISServer needs at least one ckpt path")
        self.ckpt_path = self.ckpt_paths[0]  # legacy alias for slot 0
        self.n_workers = n_workers
        self.device = device
        self.fp16 = fp16
        self.amp_dtype_name = amp_dtype_name
        self.min_batch = min_batch
        self.timeout_ms = timeout_ms
        self._proc = None
        self._handles: list = []
        # Phase 4.5: ctrl pipe + handle for parent-only commands (reload).
        # Initialized in spawn(); shutdown() closes them.
        self._ctrl_req_writer = None
        self._ctrl_resp_reader = None
        self._ctrl_handle: Optional["CISClientHandle"] = None

    def spawn(self, ready_timeout_s: float = 60.0) -> list:
        """Start CIS subprocess. Returns list of N CISClientHandle objects.

        Phase 4.5: also creates a separate CTRL pipe pair between parent and
        CIS. Parent uses ctrl pipe for reload calls; worker procs only see
        the worker pipes. This eliminates the iter-boundary race where
        worker recv_loops would steal parent's reload responses (Phase 4.4
        had this bug at iter 1+ boundaries)."""
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

        # Phase 4.5: separate CTRL pipe pair (parent <-> CIS, worker procs
        # NEVER see this). Used for parent-only reload calls.
        ctrl_req_r, ctrl_req_w = ctx.Pipe(duplex=False)   # parent writes, CIS reads
        ctrl_resp_r, ctrl_resp_w = ctx.Pipe(duplex=False)  # CIS writes, parent reads
        self._ctrl_req_writer = ctrl_req_w
        self._ctrl_resp_reader = ctrl_resp_r

        self._proc = ctx.Process(
            target=_cis_main_multi,
            args=(worker_req_readers, worker_resp_writers,
                  self.ckpt_paths, self.device, self.fp16, self.amp_dtype_name,
                  self.min_batch, self.timeout_ms,
                  ctrl_req_r, ctrl_resp_w),
            daemon=False,
        )
        self._proc.start()
        # Close the child-side pipe ends in the parent (held only by child)
        for r in worker_req_readers:
            r.close()
        for w in worker_resp_writers:
            w.close()
        ctrl_req_r.close()
        ctrl_resp_w.close()

        # Build handles WITHOUT starting their recv threads — parent-side
        # handles use the sync lock-based path (only used for occasional
        # reload calls between iters; no concurrency on parent side). The
        # async-dispatch recv-thread is started ONLY in worker procs (where
        # concurrent battles share one handle and need demuxing). See
        # _cis_worker_main, which constructs fresh handles with default
        # start_recv=True.
        self._handles = []
        for i in range(self.n_workers):
            self._handles.append(CISClientHandle(
                req_writer=worker_req_writers[i],
                resp_reader=worker_resp_readers[i],
                worker_idx=i,
                start_recv=False,  # parent-side: stay in sync mode
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

        # Phase 4.5: also consume ready on ctrl pipe + build ctrl handle.
        if not self._ctrl_resp_reader.poll(timeout=ready_timeout_s):
            raise RuntimeError(f"CIS ctrl pipe not ready in {ready_timeout_s}s")
        msg = self._ctrl_resp_reader.recv()
        if msg.get("status") != "ready":
            raise RuntimeError(f"CIS ctrl unexpected ready msg: {msg!r}")
        # Build ctrl handle: sync mode (no recv_thread). Parent uses for reloads.
        self._ctrl_handle = CISClientHandle(
            req_writer=self._ctrl_req_writer,
            resp_reader=self._ctrl_resp_reader,
            worker_idx=-1,  # sentinel: ctrl, not a worker
            start_recv=False,
        )

        return self._handles

    def reload_weights(self, weights_path: str, timeout_s: float = 60.0,
                       slot: int = 0) -> Dict[str, Any]:
        """Weight sync: signal CIS to reload from disk. Production caller
        writes weights atomically first (via _save_worker_weights_atomic in
        mp_disk_collect.py or equivalent), then invokes this. Blocks until
        reload completes; safe to start next iter's infer requests after
        this returns.

        Phase 4.5: routes through the dedicated CTRL pipe (parent <-> CIS).
        Worker procs don't see this pipe → no race with their recv_loops.
        Earlier Phase 4.4 used worker handle 0's pipe, which raced with
        worker recv_loops at iter boundaries (S53 bug). `slot` selects
        which model slot to reload (default 0 = player; slots 1..K = PFSP
        opp pool entries)."""
        if self._ctrl_handle is None:
            raise RuntimeError("CIS not spawned (no ctrl handle)")
        return self._ctrl_handle.reload(weights_path, timeout_s=timeout_s,
                                         slot=slot)

    def is_alive(self) -> bool:
        """Phase 4.6 (S54): used by orchestrator failure-detection path.
        Returns True only if CIS subprocess is spawned and running."""
        return self._proc is not None and self._proc.is_alive()

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        for h in self._handles:
            h.shutdown()
        # Phase 4.5: also signal shutdown via ctrl pipe.
        if self._ctrl_handle is not None:
            try:
                self._ctrl_handle.shutdown()
            except Exception:
                pass
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
        for p in (self._ctrl_req_writer, self._ctrl_resp_reader):
            if p is not None:
                try:
                    p.close()
                except Exception:
                    pass
        self._proc = None
        self._handles = []
        self._ctrl_handle = None
        self._ctrl_req_writer = None
        self._ctrl_resp_reader = None


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
                 fp16: bool = False, min_batch: int = 8, timeout_ms: int = 20,
                 slot: int = 0):
        # `min_batch` and `timeout_ms` are accepted for InferenceBatcher API
        # compat but not used here - CIS does the batching, not us.
        # Phase 4.3a: `slot` selects which model slot in CIS to route to.
        # slot=0 is the player slot (default for back-compat with Phase 1-3
        # tests). Worker creates one batcher per opp with slot=1..K matching
        # the orchestrator's pool slot map.
        self.handle = handle
        self.device = device
        self.fp16 = fp16
        self.slot = int(slot)
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
            lambda: self.handle.infer(np_batch, history=np_history,
                                      slot=self.slot, timeout_s=60.0),
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

    # S68 Path A1: workers are IO-bound on CIS responses; they only do
    # tiny CPU tensor ops (9-elem logits, scalar value, D-dim summary).
    # Without this, torch defaults each worker's OMP/MKL thread pool to
    # num_cpus → 60 workers × 16+ threads each can exhaust the per-user
    # nproc ulimit and trigger `libgomp: Thread creation failed`. One
    # thread per worker is correct AND eliminates the failure mode.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set earlier in process lifetime; non-fatal

    import logging
    logging.basicConfig(level=logging.WARNING)

    # Reconstruct handle from inherited pipe ends.
    # CRITICAL: do NOT start the recv_thread here. Parent's CISServer also
    # holds (read) end of the same resp pipe and uses it for reload calls
    # at iter start. If worker's recv_thread starts now, it RACES with
    # parent's reload reads — worker steals reload responses, parent gets
    # corrupted recv ("invalid load key"). Defer recv_loop start until
    # the first collect_iter cmd arrives (by then parent's reloads done).
    cis_handle = CISClientHandle(
        req_writer=cis_req_writer,
        resp_reader=cis_resp_reader,
        worker_idx=worker_id,
        start_recv=False,
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

        # First collect_iter received: parent's reload-window has closed.
        # Safe to start the recv_loop now (idempotent across iters).
        state["cis_handle"].start_recv_loop()

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
    # S68 hierarchical teambuilders: syn_config dict (or None) for
    # paired-pool team distribution (TopMixer / SynergisticMixer).
    syn_config = cmd.get("syn_config")
    rng_seed = cmd.get("rng_seed", 0)
    opp_pool = cmd["opp_pool"]
    # Phase 4.3a: orchestrator broadcasts pool_slot_map (opp_path -> slot_idx).
    # Empty/missing falls back to Phase 4.1 single-slot self-play.
    pool_slot_map = cmd.get("pool_slot_map") or {}
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
        pool_slot_map=pool_slot_map,
        syn_config=syn_config,
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
                               heartbeat_fn, liveness_state=None,
                               pool_slot_map: Optional[Dict[str, int]] = None,
                               syn_config: Optional[dict] = None):
    """CIS-routed self-play collect inside one worker.

    Differences from mp_disk's _run_collect_in_worker:
    - No model load (CIS owns the model)
    - No opp ckpt cache (CIS loads opp via reload_weights when self-play
      switches opp; orchestrator pre-loads PFSP pool slots before iter)
    - Uses CISInferenceBatcher instead of InferenceBatcher

    Phase 4.3a: `pool_slot_map` (opp_path -> slot_idx) tells the worker
    which CIS slot holds each opp's weights. Player always uses slot 0.
    Each opp gets its own batcher tagged with the right slot, so requests
    route to the correct model on the server side.

    Memory hygiene fixes (3.5b/c) + POKE_LOOP threading rule (await
    asyncio.sleep(1.5)) all carry over from mp_disk.
    """
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from rl_player import V9RLPlayer, make_self_play_opponent
    from rl_collection import _make_server
    from teams_ou import random_pool_teambuilder
    from team_generator import procedural_teambuilder, build_train_teambuilder
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

    # S68 hierarchical teambuilders (2026-06-06): build_train_teambuilder
    # returns either a ProceduralTeambuilder (legacy, no syn_config) or a
    # TopMixer (paired-pool design with syn teams). When syn_config is set,
    # the returned TB exposes yield_pair() — _play_vs_opp uses paired queues
    # to deliver matched-pool teams to both sides per battle.
    train_tb = build_train_teambuilder(
        procedural_teams_path=procedural_teams_path,
        syn_config=syn_config,
    )
    if train_tb is not None and hasattr(train_tb, 'yield_pair'):
        print(f"[cis-w{worker_id}] paired-pool teambuilder active "
              f"(syn_config={syn_config})", flush=True)

    # Phase 4.3a: build per-slot batchers. Player always slot 0; each opp
    # path gets the slot from pool_slot_map (orchestrator-assigned). If the
    # map is missing or empty (legacy callers), fall back to slot 0 for
    # everything — preserves Phase 4.1 self-play-vs-self behavior.
    pool_slot_map = pool_slot_map or {}
    player_batcher = CISInferenceBatcher(
        handle=cis_handle, device=device, fp16=fp16,
        min_batch=min(8, max_concurrent), timeout_ms=15, slot=0,
    )
    opp_batchers: Dict[str, CISInferenceBatcher] = {}
    # S67-EXT: only LOCAL opps need a CIS batcher (loaded model in slot).
    # External opps (subprocess) make their own decisions; we just challenge them.
    for opp_entry in opp_pool:
        if opp_entry.get("kind", "local") != "local":
            continue  # external — no CIS slot needed
        opp_path = opp_entry["path"]
        slot_idx = pool_slot_map.get(opp_path, 0)
        opp_batchers[opp_path] = CISInferenceBatcher(
            handle=cis_handle, device=device, fp16=fp16,
            min_batch=min(8, max_concurrent), timeout_ms=15, slot=slot_idx,
        )

    all_trajs = []
    total_w_count = total_l = total_ties = 0
    total_fft_w = total_fft_l = 0
    wr_per_opp: Dict[str, Dict[str, int]] = {}
    n_done = 0

    async def _play_vs_opp(opp_entry: dict, n_for_opp: int):
        nonlocal n_done, total_w_count, total_l, total_ties, total_fft_w, total_fft_l
        # S67-EXT CIS+EXTERNAL Tier 1: dispatch on opp kind.
        # - kind="local" (default): existing path — V9RLPlayer opponent via CIS slot
        # - kind="external_subprocess": opp is a separate Showdown user (FP/MM
        #   subprocess); use player.send_challenges(username) instead of
        #   constructing an in-process opponent.
        kind = opp_entry.get("kind", "local")
        opp_key = opp_entry.get("key") or opp_entry.get("path") or "unknown"
        batch_id = (worker_id * 100000) + (iter_n * 1000) + (hash(opp_key) % 1000)

        # S68 paired-pool setup (2026-06-06): if train_tb supports yield_pair()
        # (i.e. TopMixer or SynergisticMixer is configured), pre-fill per-side
        # team queues so both sides draw matched-source teams per battle.
        # Otherwise fall back to legacy shared-teambuilder behavior.
        _use_paired = train_tb is not None and hasattr(train_tb, 'yield_pair')
        _paired_queue_dirs = []  # collected for cleanup at end
        if _use_paired:
            import tempfile as _tempfile
            from team_generator import PairedQueueProducer, QueueTeambuilder
            _q_p1_dir = _tempfile.mkdtemp(
                prefix=f"sp_p1_w{worker_id}_i{iter_n}_b{batch_id}_"
            )
            _paired_queue_dirs.append(_q_p1_dir)
            if kind == "external_subprocess":
                _opp_team_queue = opp_entry.get("team_queue_dir")
                if _opp_team_queue:
                    PairedQueueProducer(train_tb, _q_p1_dir, _opp_team_queue).produce_all(n_for_opp)
                    # clean_on_init=False: our producer just pre-filled this
                    # queue. QueueTeambuilder default clean_on_init=True would
                    # DELETE the pairs we just pushed.
                    p1_tb = QueueTeambuilder(_q_p1_dir, clean_on_init=False)
                    p2_tb = None  # subprocess opp has no in-proc TB
                else:
                    # No opp queue available — can't pair. Fall back to legacy.
                    _use_paired = False
                    p1_tb = train_tb or random_pool_teambuilder()
                    p2_tb = train_tb or random_pool_teambuilder()
            else:
                _q_p2_dir = _tempfile.mkdtemp(
                    prefix=f"sp_p2_w{worker_id}_i{iter_n}_b{batch_id}_"
                )
                _paired_queue_dirs.append(_q_p2_dir)
                PairedQueueProducer(train_tb, _q_p1_dir, _q_p2_dir).produce_all(n_for_opp)
                # clean_on_init=False: see comment above.
                p1_tb = QueueTeambuilder(_q_p1_dir, clean_on_init=False)
                p2_tb = QueueTeambuilder(_q_p2_dir, clean_on_init=False)
        else:
            p1_tb = train_tb or random_pool_teambuilder()
            p2_tb = train_tb or random_pool_teambuilder()

        # Training player ALWAYS uses CIS player batcher (we still need NN inference)
        player = V9RLPlayer(
            batcher=player_batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=turn_cap,
            battle_format=battle_format,
            team=p1_tb,
            max_concurrent_battles=min(max_concurrent, n_for_opp),
            account_configuration=AccountConfiguration(
                f"CISw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        if kind == "local":
            opp_path = opp_entry["path"]
            # Phase 4.3a: opp goes through its assigned slot. If pool_slot_map
            # didn't list this opp, we fall back to the player batcher (preserves
            # Phase 4.1 self-play-vs-self behavior).
            # S68: opp's teambuilder is p2_tb — either a QueueTeambuilder
            # (paired mode) or shared train_tb (legacy mode).
            opp_batcher = opp_batchers.get(opp_path, player_batcher)
            opponent = V9RLPlayer(
                batcher=opp_batcher, device=device,
                reward_shaper_cfg=rs_cfg,
                temperature=1.0,
                turn_cap=turn_cap,
                battle_format=battle_format,
                team=p2_tb,
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
                print(f"[cis-w{worker_id}] timeout vs {opp_key}", flush=True)
            except Exception as e:
                print(f"[cis-w{worker_id}] error vs {opp_key}: {e}", flush=True)

        elif kind == "external_inprocess":
            # S67-EXT Tier 2: in-process external opp (e.g. PokeEnginePlayer
            # MCTS). Worker imports + constructs the opponent player directly
            # from factory_kwargs. Same battle_against flow as local — the
            # opponent just isn't NN-based (it uses its own internal MCTS).
            factory_type = opp_entry.get("factory_type", "pokeengine")
            f_kwargs = opp_entry.get("factory_kwargs", {})
            # S68: in-process opp's teambuilder is p2_tb — paired or legacy
            try:
                if factory_type == "pokeengine":
                    from pokeengine_player import PokeEnginePlayer
                    opponent = PokeEnginePlayer(
                        search_time_ms=int(f_kwargs.get("search_time_ms", 200)),
                        battle_format=battle_format,
                        team=p2_tb,
                        max_concurrent_battles=min(max_concurrent, n_for_opp),
                        account_configuration=AccountConfiguration(
                            f"CISei{worker_id}r{batch_id}", None),
                        server_configuration=srv,
                    )
                elif factory_type == "heuristic":
                    # S68 (2026-06-10) heuristic-bot adapter — in-process
                    # Player from policy_trainbots / policy_rulebots / poke_env.
                    from external_adapters import _resolve_heuristic_class
                    bot_class_name = f_kwargs.get("bot_class")
                    if not bot_class_name:
                        print(f"[cis-w{worker_id}] heuristic missing bot_class "
                              f"for {opp_key}; skipping", flush=True)
                        return
                    bot_cls = _resolve_heuristic_class(bot_class_name)
                    opponent = bot_cls(
                        battle_format=battle_format,
                        team=p2_tb,
                        max_concurrent_battles=min(max_concurrent, n_for_opp),
                        account_configuration=AccountConfiguration(
                            f"CISeh{worker_id}r{batch_id}", None),
                        server_configuration=srv,
                    )
                else:
                    print(f"[cis-w{worker_id}] unknown external_inprocess "
                          f"factory_type={factory_type} for {opp_key}; skipping",
                          flush=True)
                    return
            except ImportError as e:
                print(f"[cis-w{worker_id}] cannot import factory for {opp_key}: {e}",
                      flush=True)
                return
            except Exception as e:
                print(f"[cis-w{worker_id}] error constructing {opp_key}: {e}",
                      flush=True)
                return

            try:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=n_for_opp),
                    timeout=max(300, n_for_opp * 60),  # MCTS slower than NN
                )
            except asyncio.TimeoutError:
                print(f"[cis-w{worker_id}] timeout vs in-process {opp_key}",
                      flush=True)
            except Exception as e:
                print(f"[cis-w{worker_id}] error vs in-process {opp_key}: {e}",
                      flush=True)

        elif kind == "external_subprocess":
            # External Showdown user (FP/MM): no in-process opponent, just
            # send challenges to their username. The subprocess (running in
            # its own venv) accepts and plays.
            #
            # S67-EXT: matches legacy rl_collection._play_one_opponent's
            # protections (Session 42-44 bug fixes):
            #   1. team_queue_dir enqueue (matched-source teams)
            #   2. dispatch watchdog (5-min stall detection, 30-min hard cap)
            #   Layer 4 (force-restart stuck subprocess) is NOT here: workers
            #   in CIS subprocesses don't have direct access to external_manager.
            #   Manager's monitor_thread auto-restarts DEAD subprocesses; for
            #   STUCK subprocesses, we cancel the dispatch + skip the games
            #   this iter. Next iter may retry; if still stuck, manager
            #   eventually catches via its own checks.
            username = opp_entry["username"]
            team_queue_dir = opp_entry.get("team_queue_dir")

            # Layer 1: team_queue enqueue (if subprocess uses QueueTeambuilder).
            # Pre-enqueue n_for_opp teams so subprocess plays matched-source.
            # S68: In paired mode, the producer above already pushed P2 teams
            # to team_queue_dir as part of the paired-pool setup — skip the
            # legacy independent enqueue. In legacy mode, enqueue independently
            # from train_tb (or fallback).
            if team_queue_dir and not _use_paired:
                _enq_tb = train_tb or random_pool_teambuilder()
                try:
                    from team_generator import enqueue_team
                    for _ in range(n_for_opp):
                        try:
                            enqueue_team(team_queue_dir, _enq_tb.yield_team())
                        except Exception as e:
                            print(f"[cis-w{worker_id}] enqueue_team for {opp_key} failed: {e}",
                                  flush=True)
                            break
                except ImportError:
                    print(f"[cis-w{worker_id}] team_queue_dir set but team_generator not importable",
                          flush=True)

            # Layer 3: dispatch watchdog. Wrap send_challenges as task so we
            # can monitor progress. Cancel if stalled > stall_threshold_s,
            # absolute cap at hard_cap_s (per legacy spirit).
            stall_threshold_s = 5 * 60      # 5 min without a single battle finishing
            hard_cap_s = 30 * 60            # absolute max per opponent per iter
            poll_interval_s = 15

            challenge_task = asyncio.create_task(
                player.send_challenges(username, n_challenges=n_for_opp)
            )
            t_start = time.time()
            last_progress_t = t_start
            last_completed = 0
            try:
                while not challenge_task.done():
                    await asyncio.sleep(poll_interval_s)
                    if challenge_task.done():
                        break
                    now = time.time()
                    completed = (player.n_won_battles + player.n_lost_battles
                                 + player.n_tied_battles)
                    if completed > last_completed:
                        last_completed = completed
                        last_progress_t = now
                    stalled_s = now - last_progress_t
                    elapsed_s = now - t_start
                    if completed >= n_for_opp:
                        break  # task wrapping up; let it finish
                    if stalled_s >= stall_threshold_s:
                        print(f"[cis-w{worker_id}] external {opp_key} stalled at "
                              f"{completed}/{n_for_opp} for {int(stalled_s)}s — "
                              f"cancelling dispatch; skipping remaining "
                              f"{n_for_opp - completed} games this iter",
                              flush=True)
                        challenge_task.cancel()
                        break
                    if elapsed_s >= hard_cap_s:
                        print(f"[cis-w{worker_id}] external {opp_key} exceeded "
                              f"hard cap {hard_cap_s}s at {completed}/{n_for_opp} — "
                              f"cancelling dispatch", flush=True)
                        challenge_task.cancel()
                        break
                # Best-effort: let cancellation propagate cleanly
                try:
                    await asyncio.wait_for(challenge_task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            except Exception as e:
                print(f"[cis-w{worker_id}] error vs external {opp_key} ({username}): {e}",
                      flush=True)
                try:
                    challenge_task.cancel()
                except Exception:
                    pass
        else:
            print(f"[cis-w{worker_id}] WARN unknown opp kind={kind} for {opp_key}; "
                  f"skipping", flush=True)
            return

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
        wr_per_opp[opp_key] = {
            "w": opp_w, "g": opp_w + opp_l,
            "fft_w": opp_fft_w, "fft_l": opp_fft_l,
        }
        n_done += n_for_opp
        if liveness_state is not None:
            liveness_state["n_done"] = n_done

        # S67-ext: `opponent` is only defined for local + external_inprocess
        # paths (set inside their branches). external_subprocess opps don't
        # have a local opponent Player — the subprocess plays its own side.
        # Guard cleanup with locals() check so external_subprocess doesn't
        # raise UnboundLocalError on the post-dispatch cleanup.
        try:
            player.reset_battles()
            if 'opponent' in locals() and opponent is not None:
                opponent.reset_battles()
        except EnvironmentError:
            pass
        try:
            _cancel_listener(player)
            if 'opponent' in locals() and opponent is not None:
                _cancel_listener(opponent)
        except Exception:
            pass
        del player
        if 'opponent' in locals():
            del opponent

        # S68: clean up paired-mode temp queue dirs (only the ones we created
        # for this opp call; external opp's persistent team_queue_dir stays).
        if _paired_queue_dirs:
            import shutil as _shutil
            for _d in _paired_queue_dirs:
                try:
                    _shutil.rmtree(_d, ignore_errors=True)
                except Exception:
                    pass

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
                # Aggregate request counts across all batchers (player + opps).
                total_reqs = player_batcher._prof_total_requests
                for ob in opp_batchers.values():
                    total_reqs += ob._prof_total_requests
                print(f"[cis-w{worker_id} t+{loop_counter*15}s] "
                      f"infer_reqs={total_reqs} "
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

    # S68 thread-cap fix: bound the asyncio executor (used by
    # CISInferenceBatcher.submit for blocking pipe.send to CIS) to the
    # per-worker concurrent-battles upper bound, NOT the python default
    # min(32, cpus+4) which is 32 on 128-core pods. At 120 workers × 32
    # default threads = 3840 system threads just for asyncio executors,
    # which exhausts the per-user nproc ulimit (~4096) and causes
    # "can't start new thread" cascade. Cap = min(max_concurrent, n_games)
    # gives each worker EXACTLY enough threads for its real workload.
    # Floor of 8 ensures no over-tightening on small allocations.
    import concurrent.futures as _cf
    _max_battles = min(max_concurrent, n_games) if n_games > 0 else max_concurrent
    _exec_max = max(8, _max_battles)
    loop.set_default_executor(_cf.ThreadPoolExecutor(
        max_workers=_exec_max,
        thread_name_prefix=f"cis-w{worker_id}-exec",
    ))

    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()

    # S68 (2026-06-07): per-worker team distribution log. Only fires in
    # paired-pool mode (TopMixer/SynergisticMixer w/ selection_stats()).
    # Validates that --syn-team-pct / --top-asymmetric-rate / --syn-intra-
    # asymmetric-rate parameters are honored at runtime. One line per
    # worker per iter; grep '[TEAM-DIST]' to filter.
    if train_tb is not None and hasattr(train_tb, 'selection_stats'):
        try:
            stats = train_tb.selection_stats()
            tops = stats.get("tops", {})
            pairs = stats.get("pairs", {})
            syn_inner = stats.get("synergistic_internal", {})
            syn_sources = syn_inner.get("sources", {})
            syn_pairs = syn_inner.get("pairs", {})
            n_total_pairs = sum(pairs.values()) if pairs else 0
            n_total_yields = sum(tops.values()) if tops else 0
            n_syn_pairs = syn_pairs.get("matched", 0) + syn_pairs.get("asymmetric", 0)
            # Actual rates (compare against args' targets in train_rl.py banner)
            top_async_actual = (pairs.get("asymmetric", 0) / n_total_pairs
                                if n_total_pairs else 0.0)
            intra_async_actual = (syn_pairs.get("asymmetric", 0) / n_syn_pairs
                                  if n_syn_pairs else 0.0)
            proc_yields = tops.get("procedural", 0)
            syn_yields = tops.get("synergistic", 0)
            proc_pct = proc_yields / n_total_yields if n_total_yields else 0.0
            print(
                f"[cis-w{worker_id} TEAM-DIST iter={iter_n}] "
                f"pairs={n_total_pairs} "
                f"yields=(proc:{proc_yields} syn:{syn_yields}={proc_pct*100:.1f}%/{(1-proc_pct)*100:.1f}%) "
                f"top_pairs=(both_proc:{pairs.get('both_proc',0)} "
                f"both_syn:{pairs.get('both_syn',0)} "
                f"async:{pairs.get('asymmetric',0)}={top_async_actual*100:.1f}%) "
                f"syn_sources=({' '.join(f'{k}:{v}' for k,v in syn_sources.items())}) "
                f"intra_async={intra_async_actual*100:.1f}%",
                flush=True,
            )
        except Exception as e:
            print(f"[cis-w{worker_id}] TEAM-DIST log failed: {e}", flush=True)

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

# Phase 4.6 (S54): track timestamps of full-reset events for cap enforcement.
# A persistent failure mode that fires reset > MAX_RESETS_IN_WINDOW times in
# a 5-min sliding window aborts the run rather than spinning indefinitely.
_RESET_HISTORY: List[float] = []
_MAX_RESETS_IN_WINDOW = 5
_RESET_WINDOW_S = 300.0


class CISResetNeeded(Exception):
    """Phase 4.6 (S54): raised by _cis_wait_results_and_aggregate when a
    failure is detected (worker death, stale heartbeat, pipe EOF, or CIS
    subprocess death). Caller catches it, runs `_orchestrator_full_reset`,
    and re-dispatches the iter. Capped via _RESET_HISTORY."""
    pass


def _record_reset_and_check_cap() -> bool:
    """Append now to _RESET_HISTORY, prune old entries, return True if
    we're still under the cap (safe to retry) or False if we've reset
    too often recently (give up)."""
    now = time.time()
    _RESET_HISTORY[:] = [t for t in _RESET_HISTORY if t > now - _RESET_WINDOW_S]
    _RESET_HISTORY.append(now)
    return len(_RESET_HISTORY) <= _MAX_RESETS_IN_WINDOW


def _orchestrator_full_reset(weights_path: str, n_workers: int, device: str,
                              fp16: bool, amp_dtype_name: Optional[str],
                              min_batch: int, timeout_ms: int,
                              max_pool_size: int) -> Dict[str, Any]:
    """Phase 4.6 (S54) Option B unified recovery: tear down current CIS +
    workers, spawn fresh, restore opp slot paths from prior `current_slot_paths`.

    Triggered by `CISResetNeeded` from `_cis_wait_results_and_aggregate`.
    Cost: ~60-90s (CIS spawn + sequential K+1 slot loads + N worker spawns).
    Expected frequency: <2 events per 200-iter run (mp_disk track record:
    0 failures in 36+ iters). Returns the new `_CIS_GLOBAL` dict.

    Slot 0 is set to `weights_path` by the fresh CIS spawn; opp slots
    1..K_max start at the placeholder and are then reloaded with their
    pre-failure paths so PFSP state survives the reset."""
    global _CIS_GLOBAL

    # Snapshot current slot paths BEFORE teardown so we can restore them.
    prior_slot_paths: List[str] = list(weights_path for _ in range(max_pool_size + 1))
    if _CIS_GLOBAL is not None:
        prior_slot_paths = list(_CIS_GLOBAL.get("current_slot_paths",
                                                 prior_slot_paths))

    # Tear down old CIS + workers. Order: workers first (they need pipes
    # to CIS that we're about to close), then CIS subprocess.
    if _CIS_GLOBAL is not None:
        old_g = _CIS_GLOBAL
        _CIS_GLOBAL = None
        try:
            old_g["manager"].kill_all()
        except Exception as e:
            print(f"[cis-reset] WARN manager.kill_all: {e}", flush=True)
        try:
            old_g["server"].shutdown()
        except Exception as e:
            print(f"[cis-reset] WARN server.shutdown: {e}", flush=True)

    # Spawn fresh CIS + workers. _ensure_cis_global will set _CIS_GLOBAL.
    new_g = _ensure_cis_global(
        weights_path=weights_path, n_workers=n_workers,
        device=device, fp16=fp16, amp_dtype_name=amp_dtype_name,
        min_batch=min_batch, timeout_ms=timeout_ms,
        max_pool_size=max_pool_size,
    )

    # Restore opp slot paths. Slot 0 already correct (player weights). For
    # slots 1..K, only reload if the prior path differs from the
    # placeholder (which is `weights_path` for fresh-spawn slots).
    n_restored = 0
    for slot_idx in range(1, len(prior_slot_paths)):
        prior = prior_slot_paths[slot_idx]
        if prior == weights_path:
            continue  # placeholder, no opp was loaded here
        try:
            new_g["server"].reload_weights(prior, slot=slot_idx,
                                            timeout_s=60.0)
            new_g["current_slot_paths"][slot_idx] = prior
            n_restored += 1
        except Exception as e:
            print(f"[cis-reset] WARN slot {slot_idx} restore "
                  f"({prior}): {e}", flush=True)
    print(f"[cis-reset] full reset complete; restored {n_restored} opp "
          f"slot(s) from prior state", flush=True)
    return new_g


def shutdown_cis_workers() -> None:
    """End-of-run cleanup. Safe to call repeatedly."""
    global _CIS_GLOBAL
    if _CIS_GLOBAL is None:
        return
    g = _CIS_GLOBAL
    _CIS_GLOBAL = None

    # Shutdown workers first (they need pipes to CIS that we're about to close).
    manager = g.get("manager")
    if manager is not None:
        manager.kill_all()

    # Then shutdown CIS server.
    if g.get("server") is not None:
        try:
            g["server"].shutdown()
        except Exception:
            pass


def _ensure_cis_global(weights_path: str, n_workers: int, device: str,
                       fp16: bool, amp_dtype_name: Optional[str],
                       min_batch: int, timeout_ms: int,
                       max_pool_size: int = 16) -> Dict[str, Any]:
    """Lazy-init the CIS server + N worker procs. Returns the global dict.

    Phase 4.3a: CIS is spawned with `max_pool_size + 1` slots (slot 0 = player,
    slots 1..K_max = PFSP opp pool entries). All slots load `weights_path`
    initially; the orchestrator reloads slots 1..K with actual opp ckpts as
    the pool fills. Memory: (max_pool_size+1) × ~80MB. At default K_max=16
    that's ~1.4GB extra GPU memory, trivial on A100 80GB."""
    global _CIS_GLOBAL
    if _CIS_GLOBAL is not None:
        return _CIS_GLOBAL

    ctx = _get_mp_ctx()

    # 1. Spawn CIS server with N pipes (one per worker) and K_max+1 slots
    n_slots = max_pool_size + 1
    initial_ckpt_paths = [weights_path] * n_slots
    cis_server = CISServer(
        ckpt_path=initial_ckpt_paths, n_workers=n_workers,
        device=device, fp16=fp16, amp_dtype_name=amp_dtype_name,
        min_batch=min_batch, timeout_ms=timeout_ms,
    )
    cis_handles = cis_server.spawn(ready_timeout_s=180.0)
    print(f"[cis-orch] CIS server spawned with {n_slots} slots "
          f"(player + {max_pool_size} opp slots)", flush=True)

    # 2. Spawn N worker procs via CISWorkerManager (Phase 4.6, S54).
    # Each worker inherits its assigned CIS pipe pair from cis_handles[wid]
    # via Process(args=...) FD inheritance.
    #
    # CRITICAL (Phase 4.6 latent-bug fix): after the worker proc is spawned,
    # we MUST close the parent's copies of the CIS pipe FDs. Spawn context
    # gives the child SEPARATE FDs via reduction; the parent's copies are
    # redundant. If we leave them open, then when a worker dies, the pipe
    # inode still has parent's reader FD open → CIS's writes to the dead
    # worker's resp pipe never raise BrokenPipeError → CIS blocks once its
    # buffer fills → CISServer.shutdown() hangs → orchestrator reset hangs.
    # The pre-S54 comment at this site claimed "main can drop its references"
    # but never actually did; this surfaced under Option B's full-reset path.
    manager = CISWorkerManager(n_workers=n_workers, ctx=ctx)
    for wid in range(n_workers):
        cis_handle_for_worker = cis_handles[wid]
        manager.initial_spawn(
            worker_id=wid,
            cis_req_writer=cis_handle_for_worker.req_writer,
            cis_resp_reader=cis_handle_for_worker.resp_reader,
            pace_s=0.5,  # SemLock-race mitigation pacing
        )
        # Close parent's redundant copies. The handle's attribute references
        # still exist; subsequent send/recv on them will raise (caught by
        # the try/except in CISClientHandle.shutdown).
        try:
            cis_handle_for_worker.req_writer.close()
        except Exception:
            pass
        try:
            cis_handle_for_worker.resp_reader.close()
        except Exception:
            pass

    time.sleep(2.0)

    _CIS_GLOBAL = {
        "server": cis_server,
        "manager": manager,                # Phase 4.6: lifecycle owner
        "handles_main_view": cis_handles,  # legacy ref kept for shutdown
        "n_workers": n_workers,
        "max_pool_size": max_pool_size,
        "n_slots": n_slots,
        # current_slot_paths[s] tracks which file is loaded in slot s.
        # Slot 0 = player (reloaded each iter). Slots 1..K_max are opp
        # slots; orchestrator only reloads when path changes (vs blindly
        # reloading every iter, which would burn ~5s × K).
        "current_slot_paths": [weights_path] * n_slots,
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
    max_pool_size: int = 16,
    worker_device: Optional[str] = None,
    pfsp_max_share: float = 0.20,
    syn_config: Optional[dict] = None,
) -> Tuple[List, int, int, int, Dict, float, dict]:
    """Synchronous one-iter CIS-routed collect.

    S68: `worker_device` (default None = str(device)) overrides the device
    string sent to workers in their collect_iter cmd. Set to "cpu" to keep
    workers off the GPU entirely (avoids ~490 MB CUDA context per worker;
    60w = 30 GB saved on main GPU). CIS subprocess still uses `device`
    (cuda) for actual model inference.

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
        max_pool_size=max_pool_size,
    )

    # Phase 4.6 (S54) Option B: dispatch + wait + retry-on-reset loop.
    # On any failure detected by _cis_wait_results_and_aggregate (worker
    # death, stale heartbeat, pipe EOF, CIS death), we tear down + re-spawn
    # everything and re-dispatch the iter from zero. Cap retries via
    # _record_reset_and_check_cap to avoid spinning on persistent failures.
    MAX_RESET_ATTEMPTS = 2  # initial + 2 retries = 3 total attempts
    last_reset_reason: Optional[str] = None
    for attempt in range(MAX_RESET_ATTEMPTS + 1):
        # 3. Slot routing: reload slot 0 every iter (player weights changed
        # via PPO update); reload opp slots only if path changed since last
        # iter. After a reset, _orchestrator_full_reset already restored opp
        # slot paths, so this becomes mostly no-op on retry.
        n_slots = g["n_slots"]
        cur_paths = g["current_slot_paths"]
        pool_slot_map: Dict[str, int] = {}
        # S67-EXT: only LOCAL opps get CIS slots (externals don't need a
        # loaded model). Iterate snapshot_pool, allocate slots only to
        # str entries (legacy local paths). Dict entries (external) are
        # routed via send_challenges later, no slot needed.
        slot_idx = 1
        for item in snapshot_pool[:max_pool_size]:
            if isinstance(item, str):
                pool_slot_map[item] = slot_idx
                slot_idx += 1
            # else: external dict — skip slot allocation

        # Reload slot 0 (player) every iter — model state changes after each
        # PPO update.
        try:
            g["server"].reload_weights(weights_path, slot=0, timeout_s=60.0)
            cur_paths[0] = weights_path
        except Exception as e:
            print(f"[cis-orch] WARN slot 0 reload at iter {iter_n}: {e}",
                  flush=True)

        # Reload opp slots whose path changed.
        n_opp_reloaded = 0
        for opp_path, slot_idx in pool_slot_map.items():
            if slot_idx >= n_slots:
                print(f"[cis-orch] WARN opp slot {slot_idx} exceeds "
                      f"n_slots={n_slots}; raise --max-pool-size", flush=True)
                continue
            if cur_paths[slot_idx] == opp_path:
                continue  # already loaded
            try:
                g["server"].reload_weights(opp_path, slot=slot_idx,
                                            timeout_s=60.0)
                cur_paths[slot_idx] = opp_path
                n_opp_reloaded += 1
            except Exception as e:
                print(f"[cis-orch] WARN slot {slot_idx} reload "
                      f"({opp_path}): {e}", flush=True)
        if n_opp_reloaded:
            print(f"[cis-orch] iter {iter_n}: reloaded {n_opp_reloaded} opp "
                  f"slot(s); pool_size={len(pool_slot_map)}", flush=True)

        # 4. Build opp_pool message + dispatch to all alive workers.
        # S58 (F): each worker receives a SINGLE opp via _allocate_opps_to_workers,
        # eliminating in-process asyncio.gather contention that produced 50-70%
        # ghost-ties on the prior 1600g run.
        opp_pool_full = _build_opp_pool_msg(snapshot_pool, win_rates)
        manager = g["manager"]
        alive_wids = manager.alive_worker_ids()
        n_alive = len(alive_wids)
        if n_alive == 0:
            raise RuntimeError(f"[cis-orch] iter {iter_n}: no alive workers")
        assignments = _allocate_opps_to_workers(opp_pool_full, n_alive, n_games,
                                                 max_share=pfsp_max_share)

        # Log per-iter distribution for visibility (one line, sorted by games).
        opp_dist: Dict[str, List[int]] = {}
        for a in assignments:
            if a["opp"] is None:
                continue
            # S67-EXT: use 'key' (canonical id), fallback to 'path' for backward compat
            opp_dist.setdefault(
                a["opp"].get("key") or a["opp"].get("path", "unknown"),
                []
            ).append(a["n_games"])
        dist_summary = ", ".join(
            f"{os.path.basename(p)}={sum(g)}g/{len(g)}w"
            for p, g in sorted(opp_dist.items(), key=lambda kv: -sum(kv[1]))
        )
        print(f"[cis-orch] iter {iter_n}: {n_alive} workers, "
              f"{len(opp_dist)} active opps -> {dist_summary}", flush=True)

        cmds_sent: List[int] = []
        t_start = time.time()
        # F4 helper: extract port from a server's websocket_url
        # (ws://host:port/showdown/websocket → port int).
        def _server_port(srv):
            url = getattr(srv, 'websocket_url', None) or str(srv)
            try:
                return int(url.split(':')[2].split('/')[0])
            except (IndexError, ValueError):
                return None
        # Pre-compute port→server map for fast lookup.
        port_to_srv = {_server_port(s): s for s in server_pool if _server_port(s) is not None}

        for i, wid in enumerate(alive_wids):
            assignment = assignments[i]
            worker_n = assignment["n_games"]
            opp_msg = assignment["opp"]

            # Default: worker bound to server_pool[i % N] (round-robin).
            srv_idx = i % len(server_pool)
            srv = server_pool[srv_idx]

            # F4 multi-port routing: if opp is a multi-instance external
            # subprocess, pick a concrete instance for this worker and OVERRIDE
            # the worker's server to match that instance's port. Workers and
            # MM instances must co-locate on the same battle_server.js process
            # (different ports = different battle_server processes with no
            # shared state). Routing: round-robin instance by worker index.
            if opp_msg and opp_msg.get("kind") == "external_subprocess":
                inst_usernames = opp_msg.get("instance_usernames")
                inst_ports = opp_msg.get("instance_ports")
                inst_queues = opp_msg.get("instance_team_queue_dirs")
                if inst_usernames and inst_ports:
                    inst_idx = i % len(inst_usernames)
                    chosen_port = inst_ports[inst_idx]
                    chosen_srv = port_to_srv.get(chosen_port)
                    if chosen_srv is not None:
                        srv = chosen_srv
                    # Bake the instance's username + queue into the per-worker
                    # opp dict. Strip instance lists (only needed at routing).
                    opp_msg = dict(opp_msg)
                    opp_msg["username"] = inst_usernames[inst_idx]
                    opp_msg["team_queue_dir"] = (
                        inst_queues[inst_idx] if inst_queues else None
                    )
                    opp_msg.pop("instance_usernames", None)
                    opp_msg.pop("instance_team_queue_dirs", None)
                    opp_msg.pop("instance_ports", None)

            single_opp_msg = [opp_msg] if opp_msg is not None else []
            srv_url = getattr(srv, 'websocket_url', None) or str(srv)
            cmd = {
                "cmd": "collect_iter",
                "iter_n": iter_n,
                "n_games": worker_n,
                "max_concurrent": max_concurrent,
                "server_url": srv_url,
                "opp_pool": single_opp_msg,
                "pool_slot_map": pool_slot_map,
                "temp_range": list(temp_range),
                "opp_temp_range": list(temp_range),
                "fp16": fp16,
                "rs_cfg": reward_shaper_cfg,
                "turn_cap": turn_cap,
                "battle_format": battle_format,
                "procedural_teams_path": procedural_teams_path,
                "syn_config": syn_config,
                "device": worker_device if worker_device else str(device),
                "opponent_device": opponent_device,
                "rng_seed": rng_seed,
                "amp_dtype": amp_dtype,
            }
            manager.ctrl_pipes[wid].send(cmd)
            cmds_sent.append(wid)
            # Stagger same as mp_disk - reduces I/O burst at iter boundary.
            if i + 1 < n_alive:
                time.sleep(0.25)

        # 5+6. Wait for results + aggregate. Raises CISResetNeeded on
        # detected failure → caught below → full reset + retry.
        try:
            return _cis_wait_results_and_aggregate(
                g=g, cmds_sent=cmds_sent, iter_n=iter_n, n_games=n_games,
                n_alive=n_alive, n_workers=n_workers, t_start=t_start,
            )
        except CISResetNeeded as e:
            if attempt >= MAX_RESET_ATTEMPTS:
                raise RuntimeError(
                    f"[cis-orch] iter {iter_n}: gave up after "
                    f"{MAX_RESET_ATTEMPTS + 1} attempts. Last reason: {e}"
                ) from e
            if not _record_reset_and_check_cap():
                raise RuntimeError(
                    f"[cis-orch] iter {iter_n}: reset cap exceeded "
                    f"({_MAX_RESETS_IN_WINDOW} resets in last "
                    f"{_RESET_WINDOW_S:.0f}s) — likely persistent failure "
                    f"mode. Last reason: {e}"
                ) from e
            print(f"[cis-orch] iter {iter_n}: reset attempt "
                  f"{attempt + 1}/{MAX_RESET_ATTEMPTS + 1}: {e}", flush=True)
            last_reset_reason = str(e)
            g = _orchestrator_full_reset(
                weights_path=weights_path, n_workers=n_workers,
                device=str(device), fp16=fp16, amp_dtype_name=amp_dtype,
                min_batch=cis_min_batch, timeout_ms=cis_timeout_ms,
                max_pool_size=max_pool_size,
            )
    # Should be unreachable — final attempt either returned or raised.
    raise RuntimeError(
        f"[cis-orch] iter {iter_n}: unreachable retry-loop exit. "
        f"Last reason: {last_reset_reason}")


def _cis_wait_results_and_aggregate(
    g: Dict[str, Any], cmds_sent: List[int], iter_n: int,
    n_games: int, n_alive: int, n_workers: int, t_start: float,
) -> Tuple[List, int, int, int, int, str, float, Dict]:
    """Watchdog loop + trajectory aggregation. Used by both sync orchestrator
    (mp_centralized_collect_sync) and bg orchestrator (CISBgCollector.join).
    Same return-tuple shape as mp_disk_collect_sync.

    Phase 4.6 (S54) Option B: on ANY failure (worker death, stale heartbeat,
    pipe EOF, CIS subprocess death) we raise `CISResetNeeded`. The caller
    runs `_orchestrator_full_reset` and re-dispatches the iter from zero.
    No partial trajectories ever reach the PPO update — binary outcome:
    iter completes cleanly OR iter is fully re-collected after reset."""
    from multiprocessing.connection import wait as mp_wait
    import gzip as _gzip
    import pickle as _pickle

    manager = g["manager"]
    cis_server = g["server"]

    # S58 defensive: filter cmds_sent to wids that actually have result_pipe
    # entries in the current manager. Background: on the retry-after-reset
    # path (line ~2863), self._cmds_sent is supposed to be refreshed from
    # _reload_slots_and_dispatch BEFORE the second _cis_wait call. Under
    # repeated resets the bg-collector path was observed to enter this
    # function with cmds_sent containing wids that aren't in
    # manager.result_pipes (KeyError 0 in prod, S58 iter 41 attempt 2).
    # Root cause not fully pinned — likely a race between the bg-collector's
    # reset path and concurrent _CIS_GLOBAL mutation, or an exception in
    # _reload_slots_and_dispatch that left self._cmds_sent stale. Until we
    # have ironclad RCA, drop stale wids defensively and surface the warn
    # so any future occurrence is logged with full context. Dropped wids
    # can't deliver results anyway (no pipe to recv from), so this is
    # behaviorally equivalent to "wait for the workers we actually have."
    valid_cmds = [wid for wid in cmds_sent if wid in manager.result_pipes]
    if len(valid_cmds) != len(cmds_sent):
        stale = set(cmds_sent) - set(manager.result_pipes.keys())
        print(f"[cis-orch] WARN iter {iter_n}: cmds_sent had stale wids "
              f"{sorted(stale)} not in manager.result_pipes "
              f"(have {sorted(manager.result_pipes.keys())}); "
              f"filtered {len(cmds_sent)} -> {len(valid_cmds)} wids",
              flush=True)
    cmds_sent = valid_cmds
    if not cmds_sent:
        # All wids stale → no useful workers. Trigger another reset.
        raise CISResetNeeded(
            f"iter {iter_n}: all cmds_sent wids stale after filter; "
            f"manager.result_pipes={sorted(manager.result_pipes.keys())}")

    expected = len(cmds_sent)
    received = 0
    results: List[dict] = []
    expected_collect_s = max(300.0, n_games / max(n_alive, 1) * 2.0)
    # S58: bumped multiplier 4.0 -> 8.0. At 1600g/8w, expected_collect_s=400s,
    # so 4.0× gave ~1600s deadline. Prod 8w iter 40 first-iter cold start
    # (CIS spawn + 6-slot model loads + 8 worker spawns + WS setup) takes
    # ~60-90s, pushing total collect to ~1700-1800s and missing the 1600s
    # deadline on attempts 1 AND 2. The validation 8w run scraped under by
    # chance. 8.0× gives ~3200s = 53 min, comfortable absorbance for cold
    # start at any reasonable worker count. Steady-state iters complete in
    # ~1200s so the higher multiplier never actually delays a clean iter —
    # it only avoids spurious resets on cold starts.
    deadline = time.time() + 8.0 * expected_collect_s

    pipes_to_wid = {manager.result_pipes[wid]: wid for wid in cmds_sent}

    while received < expected and time.time() < deadline:
        # Phase 4.6 (S54) Test 6b finding: CIS subprocess can die mid-iter
        # WITHOUT triggering worker errors — workers' inference submits hit
        # BrokenPipe but V9RLPlayer falls back to default actions, battles
        # complete with 0 trajectories, workers report `status: done` with
        # valid W/L. The wait loop must check CIS aliveness explicitly to
        # detect this; relying on worker error reports misses it.
        if not cis_server.is_alive():
            raise CISResetNeeded(
                f"iter {iter_n}: cis_subprocess_dead (mid-collect)")
        ready = mp_wait(list(pipes_to_wid.keys()), timeout=10.0)
        if not ready:
            # No worker reported in 10s — check worker liveness too.
            # ANY unhealthy condition triggers full reset (Option B).
            unhealthy = manager.health_check()
            if unhealthy:
                raise CISResetNeeded(
                    f"iter {iter_n}: workers={unhealthy}")
            continue

        for conn in ready:
            try:
                msg = conn.recv()
            except (EOFError, OSError) as e:
                # Pipe closed — worker died unexpectedly. Trigger reset.
                wid = pipes_to_wid.get(conn)
                raise CISResetNeeded(
                    f"iter {iter_n}: worker {wid} pipe "
                    f"{type(e).__name__}: {e}")
            status = msg.get("status")
            if status == "heartbeat":
                manager.last_heartbeat[msg.get("worker_id")] = time.time()
                continue
            received += 1
            if status == "done":
                results.append(msg)
            elif status == "error":
                # Worker reported a Python exception. Trigger reset rather
                # than continue with degraded data — Option B contract is
                # binary (clean iter or full reset).
                raise CISResetNeeded(
                    f"iter {iter_n}: worker {msg.get('worker_id')} "
                    f"reported error: {msg.get('exc_msg')}")

    # Deadline reached without CISResetNeeded means workers are sending
    # heartbeats but not finishing. That's degenerate — treat as reset.
    if received < expected:
        raise CISResetNeeded(
            f"iter {iter_n}: deadline reached, received={received}/{expected}")

    # Final aliveness check: even if all workers reported "done", CIS might
    # have died mid-iter and workers completed with default actions. The
    # responses look valid (W/L counts present) but trajectories are empty.
    # If CIS is dead now, treat the iter as failed → reset path.
    if not cis_server.is_alive():
        raise CISResetNeeded(
            f"iter {iter_n}: cis_subprocess_dead (post-collect, "
            f"trajectories may be invalid)")

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

    # Cleanup old traj files (keep last 2 iters).
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


class CISBgCollector:
    """Phase 4.3b: Background-mode CIS collect. Mirrors MPDiskBgCollector
    interface (start, join, running) so train_rl.py's pipeline path can
    use it as drop-in.

    Re-enables the bg overlap that was no-op'd after the mp+pipeline GPU
    contention deadlock (see train_rl.py:_start_background_collection
    pre-Phase-4.3b). The CIS low-priority CUDA stream (Phase 4.2) lets
    worker forwards make progress on the GPU during main's
    optimizer.step(), while main gets first-priority access. No more
    deadlock; collect actually overlaps with update.

    Semantics (same as MPDiskBgCollector): workers always use the LAST
    FINALIZED weights at iter boundary. Iter K's collect uses weights
    from iter K-1's update (off-by-1 stale, identical to BackgroundCollector
    and MPDiskBgCollector — PPO is robust to this 1-iter staleness).
    """

    def __init__(self):
        self._kicked_off = False
        self._iter_n: Optional[int] = None
        self._cmds_sent: List[int] = []
        self._t_start = 0.0
        self._n_games = 0
        self._n_alive = 0
        self._n_workers = 0
        self._g: Optional[Dict[str, Any]] = None
        # Phase 4.6 (S54): retain dispatch context so .join() can re-dispatch
        # on reset (Option B unified path). All fields populated by .start();
        # cleared by .join().
        self._dispatch_ctx: Optional[Dict[str, Any]] = None

    @property
    def running(self) -> bool:
        return self._kicked_off

    def _reload_slots_and_dispatch(self, g: Dict[str, Any]) -> List[int]:
        """Phase 4.6 (S54) factored helper: slot reload + dispatch
        collect_iter to all alive workers. Used by both initial start()
        and .join()'s retry-on-reset path. Returns cmds_sent."""
        ctx = self._dispatch_ctx
        assert ctx is not None
        weights_path = ctx["weights_path"]
        snapshot_pool = ctx["snapshot_pool"]
        max_pool_size = ctx["max_pool_size"]
        iter_n = ctx["iter_n"]

        # Slot reload (slot 0 every iter; opp slots only if path changed).
        n_slots = g["n_slots"]
        cur_paths = g["current_slot_paths"]
        pool_slot_map: Dict[str, int] = {}
        # S67-EXT: only LOCAL opps need CIS slots (same logic as orchestrator path above)
        _slot_idx = 1
        for item in snapshot_pool[:max_pool_size]:
            if isinstance(item, str):
                pool_slot_map[item] = _slot_idx
                _slot_idx += 1

        try:
            g["server"].reload_weights(weights_path, slot=0, timeout_s=60.0)
            cur_paths[0] = weights_path
        except Exception as e:
            print(f"[cis-bg] WARN slot 0 reload at iter {iter_n}: {e}",
                  flush=True)

        n_opp_reloaded = 0
        for opp_path, slot_idx in pool_slot_map.items():
            if slot_idx >= n_slots:
                print(f"[cis-bg] WARN opp slot {slot_idx} exceeds "
                      f"n_slots={n_slots}; raise --max-pool-size", flush=True)
                continue
            if cur_paths[slot_idx] == opp_path:
                continue
            try:
                # Phase 4.6 (S54) bug fix: route reload through CISServer's
                # ctrl handle, NOT handles_main_view[0]. Pre-4.6 bg path
                # used the racy pre-4.5 pattern (§B13).
                g["server"].reload_weights(opp_path, slot=slot_idx,
                                            timeout_s=60.0)
                cur_paths[slot_idx] = opp_path
                n_opp_reloaded += 1
            except Exception as e:
                print(f"[cis-bg] WARN slot {slot_idx} reload "
                      f"({opp_path}): {e}", flush=True)
        if n_opp_reloaded:
            print(f"[cis-bg] iter {iter_n}: reloaded {n_opp_reloaded} opp "
                  f"slot(s); pool_size={len(pool_slot_map)}", flush=True)

        # Build opp_pool message + dispatch (alive-only).
        opp_pool_msg = _build_opp_pool_msg(snapshot_pool, ctx["win_rates"])
        manager = g["manager"]
        alive_wids = manager.alive_worker_ids()
        n_alive = len(alive_wids)
        if n_alive == 0:
            raise RuntimeError(f"[cis-bg] iter {iter_n}: no alive workers")
        n_games = ctx["n_games"]

        # S58 (F): one opp per worker. See _allocate_opps_to_workers docstring
        # for rationale. Mirrors the regular orchestrator dispatch path.
        assignments = _allocate_opps_to_workers(opp_pool_msg, n_alive, n_games,
                                                 max_share=ctx.get("pfsp_max_share", 0.20))
        opp_dist: Dict[str, List[int]] = {}
        for a in assignments:
            if a["opp"] is None:
                continue
            # S67-EXT: use 'key' (canonical id), fallback to 'path' for backward compat
            opp_dist.setdefault(
                a["opp"].get("key") or a["opp"].get("path", "unknown"),
                []
            ).append(a["n_games"])
        dist_summary = ", ".join(
            f"{os.path.basename(p)}={sum(g)}g/{len(g)}w"
            for p, g in sorted(opp_dist.items(), key=lambda kv: -sum(kv[1]))
        )
        print(f"[cis-bg] iter {iter_n}: {n_alive} workers, "
              f"{len(opp_dist)} active opps -> {dist_summary}", flush=True)

        cmds_sent: List[int] = []
        for i, wid in enumerate(alive_wids):
            assignment = assignments[i]
            worker_n = assignment["n_games"]
            single_opp_msg = [assignment["opp"]] if assignment["opp"] is not None else []
            srv_idx = i % len(ctx["server_pool"])
            srv = ctx["server_pool"][srv_idx]
            srv_url = getattr(srv, 'websocket_url', None) or str(srv)
            cmd = {
                "cmd": "collect_iter",
                "iter_n": iter_n,
                "n_games": worker_n,
                "max_concurrent": ctx["max_concurrent"],
                "server_url": srv_url,
                "opp_pool": single_opp_msg,
                "pool_slot_map": pool_slot_map,
                "temp_range": list(ctx["temp_range"]),
                "opp_temp_range": list(ctx["temp_range"]),
                "fp16": ctx["fp16"],
                "rs_cfg": ctx["rs_cfg"],
                "turn_cap": ctx["turn_cap"],
                "battle_format": ctx["battle_format"],
                "procedural_teams_path": ctx["procedural_teams_path"],
                "syn_config": ctx.get("syn_config"),
                "device": ctx.get("worker_device_str") or ctx["device_str"],
                "opponent_device": ctx["opponent_device"],
                "rng_seed": ctx["rng_seed"],
                "amp_dtype": ctx["amp_dtype"],
            }
            manager.ctrl_pipes[wid].send(cmd)
            cmds_sent.append(wid)
            if i + 1 < n_alive:
                time.sleep(0.25)

        self._n_alive = n_alive
        return cmds_sent

    def start(self, model, device, server_pool, snapshot_pool, args_dict,
              win_rates=None, iter_n: int = 0):
        """Kick off CIS collect for the next iter. Returns immediately;
        results retrieved via .join(). Same args_dict shape as
        MPDiskBgCollector.start."""
        n_workers = args_dict.get("n_workers", 8)
        n_games = args_dict["games_per_iter"]
        fp16 = args_dict["fp16"]
        amp_dtype = args_dict.get("amp_dtype")
        max_pool_size = args_dict.get("max_pool_size", 16)
        cis_min_batch = args_dict.get("cis_min_batch", 8)
        cis_timeout_ms = args_dict.get("cis_timeout_ms", 15)
        rng_seed = args_dict.get("rng_seed") or random.randint(0, 1_000_000)

        if device.type == "cpu":
            raise ValueError("CISBgCollector not supported on CPU.")

        # Save weights for CIS to load
        weights_path = _save_weights_atomic_for_cis(model, iter_n)

        # Init or reuse CIS global
        g = _ensure_cis_global(
            weights_path=weights_path, n_workers=n_workers,
            device=str(device), fp16=fp16, amp_dtype_name=amp_dtype,
            min_batch=cis_min_batch, timeout_ms=cis_timeout_ms,
            max_pool_size=max_pool_size,
        )

        # Phase 4.6: stash all dispatch context so .join() can re-dispatch
        # on a reset event without needing the original args back.
        self._dispatch_ctx = {
            "weights_path": weights_path,
            "snapshot_pool": snapshot_pool,
            "win_rates": win_rates,
            "iter_n": iter_n,
            "n_workers": n_workers,
            "n_games": n_games,
            "max_concurrent": args_dict["max_concurrent"],
            "fp16": fp16,
            "rs_cfg": args_dict["rs_cfg"],
            "temp_range": args_dict["temp_range"],
            "opponent_device": args_dict.get("opponent_device", str(device)),
            "turn_cap": args_dict.get("turn_cap", 300),
            "battle_format": args_dict.get("battle_format", "gen9ou"),
            "procedural_teams_path": args_dict.get("teambuilder_path"),
            "syn_config": args_dict.get("syn_config"),
            "device_str": str(device),
            "device_type": device.type,
            # S68 worker-cpu mode: if args_dict["worker_device_str"] is set,
            # workers get that device (typically "cpu") instead of "cuda".
            # CIS subprocess + main proc still use `device` for inference.
            "worker_device_str": (args_dict.get("worker_device_str") or str(device)),
            # S68 Path B-hybrid: per-opp max share cap (default 0.20).
            "pfsp_max_share": args_dict.get("pfsp_max_share", 0.20),
            "amp_dtype": amp_dtype,
            "max_pool_size": max_pool_size,
            "cis_min_batch": cis_min_batch,
            "cis_timeout_ms": cis_timeout_ms,
            "rng_seed": rng_seed,
            "server_pool": server_pool,
        }

        cmds_sent = self._reload_slots_and_dispatch(g)

        # Track state for join().
        self._g = g
        self._cmds_sent = cmds_sent
        self._iter_n = iter_n
        self._n_games = n_games
        self._n_workers = n_workers
        self._t_start = time.time()
        self._kicked_off = True

    def join(self) -> Optional[Tuple]:
        """Block until workers finish; return aggregated result tuple.

        Phase 4.6 (S54) Option B: on CISResetNeeded, runs full reset +
        re-dispatch the iter from zero. Capped retries via
        _record_reset_and_check_cap to avoid spinning."""
        if not self._kicked_off:
            return None
        ctx = self._dispatch_ctx
        assert ctx is not None
        MAX_RESET_ATTEMPTS = 2
        last_reset_reason: Optional[str] = None
        for attempt in range(MAX_RESET_ATTEMPTS + 1):
            try:
                result = _cis_wait_results_and_aggregate(
                    g=self._g, cmds_sent=self._cmds_sent,
                    iter_n=self._iter_n, n_games=self._n_games,
                    n_alive=self._n_alive, n_workers=self._n_workers,
                    t_start=self._t_start,
                )
                self._kicked_off = False
                self._dispatch_ctx = None
                return result
            except CISResetNeeded as e:
                if attempt >= MAX_RESET_ATTEMPTS:
                    self._kicked_off = False
                    self._dispatch_ctx = None
                    raise RuntimeError(
                        f"[cis-bg] iter {self._iter_n}: gave up after "
                        f"{MAX_RESET_ATTEMPTS + 1} attempts. Last reason: {e}"
                    ) from e
                if not _record_reset_and_check_cap():
                    self._kicked_off = False
                    self._dispatch_ctx = None
                    raise RuntimeError(
                        f"[cis-bg] iter {self._iter_n}: reset cap exceeded "
                        f"({_MAX_RESETS_IN_WINDOW} resets in last "
                        f"{_RESET_WINDOW_S:.0f}s) — likely persistent "
                        f"failure mode. Last reason: {e}"
                    ) from e
                print(f"[cis-bg] iter {self._iter_n}: reset attempt "
                      f"{attempt + 1}/{MAX_RESET_ATTEMPTS + 1}: {e}",
                      flush=True)
                last_reset_reason = str(e)
                self._g = _orchestrator_full_reset(
                    weights_path=ctx["weights_path"],
                    n_workers=ctx["n_workers"],
                    device=ctx["device_str"],
                    fp16=ctx["fp16"],
                    amp_dtype_name=ctx["amp_dtype"],
                    min_batch=ctx["cis_min_batch"],
                    timeout_ms=ctx["cis_timeout_ms"],
                    max_pool_size=ctx["max_pool_size"],
                )
                self._cmds_sent = self._reload_slots_and_dispatch(self._g)
                self._t_start = time.time()
        # Unreachable.
        self._kicked_off = False
        self._dispatch_ctx = None
        raise RuntimeError(
            f"[cis-bg] iter {self._iter_n}: unreachable retry-loop exit. "
            f"Last reason: {last_reset_reason}")


def _entry_key(item) -> str:
    """S67-EXT: canonical key for an opp entry (str or dict).

    For local (str): the path IS the key.
    For external dict: explicit "key" field.
    Used as the dict key for stats (win_rates, pool_slot_map, etc).
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("key") or item.get("path") or "unknown"
    return str(item)


def _entry_is_local(item) -> bool:
    """S67-EXT: is this opp entry a local checkpoint (vs external)?"""
    if isinstance(item, str):
        return True
    if isinstance(item, dict):
        return item.get("kind", "local") == "local"
    return False


def _build_opp_pool_msg(snapshot_pool, win_rates) -> List[dict]:
    """Build PFSP-weighted opp_pool list for the cmd dict. Same shape as
    mp_disk_collect's helper of the same name.

    S67-EXT: handles both legacy string entries (local paths) and dict
    entries (external opps with kind/key/username fields). Output dict
    always has 'key' field; 'path' only for local entries; 'kind' and
    external fields propagated for non-local.
    """
    out = []
    for item in snapshot_pool:
        key = _entry_key(item)
        wr_entry = (win_rates or {}).get(key, {})
        if isinstance(wr_entry, dict):
            w = wr_entry.get("w", 0)
            g = max(wr_entry.get("g", 0), 1)
            wr = w / g
        elif isinstance(wr_entry, list) and len(wr_entry) == 2:
            # EMA format: [ema_wins, eff_games]
            w, g = wr_entry
            wr = w / max(g, 1)
        else:
            wr = 0.5
        weight = (1.0 - wr) ** 2
        if isinstance(item, str):
            out.append({"kind": "local", "key": item, "path": item,
                        "wr": wr, "weight": weight})
        elif isinstance(item, dict):
            # Pass through external dict, augment with wr/weight
            entry = dict(item)  # copy
            entry["wr"] = wr
            entry["weight"] = weight
            out.append(entry)
        else:
            raise ValueError(f"Unknown opp entry type: {type(item).__name__}")
    return out


# S67-ext speed-aware allocation (2026-05-28): per-game wall-time estimates
# used to size worker count per opp. ONLY activated when at least one opp
# crosses _SLOW_OPP_THRESHOLD_S (e.g., mcts-hard at ~30s/game). Otherwise the
# allocator falls through to the legacy games-balanced greedy that produces
# the clean even distribution (3 workers per opp for typical 30w/10opp iter).
# Per-game estimates are calibrated from smoke v9 (mcts-hard worker wall =
# 23 min for 53 games ≈ 26s/game) and only need ordinal accuracy.
_SLOW_OPP_THRESHOLD_S = 20.0
# Within speed-aware mode, cap self-play opps at this many workers each. SP
# wall is dominated by battle length (~10 min), not per-game cost — adding
# workers to SP doesn't reduce wall because each worker already runs 50+
# concurrent battles via asyncio.gather, sharing CIS inference. Capping SP
# frees workers for slow opps (mcts-hard) that DO scale linearly with worker
# count. Only applied when speed-aware mode is active; in normal iters SP
# gets the usual 3 workers per opp like everyone else.
_SP_CAP_IN_SPEED_AWARE = 2


def _estimate_per_game_seconds(opp_msg: dict) -> float:
    """Rough s/game estimate by opp kind. Used to decide whether speed-aware
    allocation should activate, and (when activated) to weight worker count
    per opp by wall-time cost rather than game count."""
    kind = opp_msg.get("kind", "local")
    if kind == "external_inprocess":
        f_kwargs = opp_msg.get("factory_kwargs", {})
        factory_type = f_kwargs.get("factory_type", "pokeengine")
        if factory_type == "heuristic":
            # In-process Python heuristic bot — fast (~3s/game like
            # local self-play, no MCTS search budget).
            return 3.0
        # MCTS via poke-engine: ~25 turns × search_time_ms + ~5s overhead
        # (state-build, network, choose_move serialization).
        st_ms = int(f_kwargs.get("search_time_ms", 200))
        return st_ms * 0.025 + 5.0
    if kind == "external_subprocess":
        key = (opp_msg.get("key") or "").lower()
        if key.startswith("mm-") or "metamon" in key:
            return 5.0  # Metamon: NN inference in subprocess, fast
        if key.startswith("fp-") or "foul" in key or "foulplay" in key:
            return 20.0  # Foul Play subprocess: MCTS too, but full strategy stack
        return 10.0  # generic external subprocess
    # kind == "local" (self-play snapshots, GPU inference via shared CIS)
    return 10.0


def _allocate_opps_to_workers(
    opp_pool_msg: List[dict],
    n_workers: int,
    total_games: int,
    max_share: float = 0.20,
) -> List[dict]:
    """S58 (F): orchestrator-side opp -> worker assignment. One opp per worker.

    Background: the prior flow had every worker run all opps in parallel via
    asyncio.gather() inside its event loop. Under PFSP-weighted allocations
    like [100, 25, 25, 25, 25] games per worker, the smaller batches starved
    under shared scheduling and hit their per-batch asyncio.wait_for timeouts
    (max(300, n_for_opp*30)s) before completing. Result: ~50-70% of games
    per iter became battle.won=None "ghost-ties" (S58 1600g run, iters 40-48).

    This matches the Metamon self-play architecture (verified in
    metamon_ref/metamon/rl/self_play/launch_models.py — one subprocess.Popen
    per username). We keep our orchestrator-side dispatch, but each worker
    now receives a single opp and plays only that opp's games. Worker's
    existing asyncio.gather() runs one coro → no contention.

    Allocation:
      - PFSP-weighted target games per opp computed from .weight field.
      - n_workers >= n_opps: every opp gets >=1 worker; extras go greedy by
        games/workers ratio (the opp most under-served).
      - n_workers <  n_opps: top-N by weight get workers; skipped opps'
        games redistribute proportionally to included opps. Skipped opps
        will rise in PFSP weight next iter (less played => higher weight).

    Returns a list of length n_workers; each entry is
        {"opp": <opp_msg_dict_or_None>, "n_games": <int>}.
    """
    n_opps = len(opp_pool_msg)
    if n_opps == 0 or n_workers == 0:
        return [{"opp": None, "n_games": 0} for _ in range(n_workers)]

    # Step 1: PFSP-weighted target games per opp.
    weights = [max(o.get("weight", 1.0), 1e-6) for o in opp_pool_msg]
    total_w = sum(weights)
    target = [int(round(total_games * w / total_w)) for w in weights]
    drift = total_games - sum(target)
    if drift != 0:
        target[weights.index(max(weights))] += drift

    # Step 1b: S68 Path B-hybrid (2026-05-31) — cap any single opp at
    # `max_share` fraction of total_games (default 0.20). PFSP-weighted
    # allocation can pathologically concentrate games on one opp at our
    # actor scale (~30 workers): e.g. mirror match (default WR=0.5,
    # weight=0.25) was observed taking 36% of games while 9 other opps
    # split the rest. The cap preserves PFSP's "harder=more" signal within
    # the cap, but prevents any single opp from dominating the iter and
    # starving the gradient signal from less-played opps. Excess from
    # capped opps redistributes by weight to uncapped opps.
    #
    # Disable by passing max_share >= 1.0 (legacy unbounded behavior).
    # See [[s68-path-b-design]] memo + AlphaStar/OpenAI Five comparison.
    if 0.0 < max_share < 1.0 and n_opps >= 2:
        cap = max(1, int(total_games * max_share))
        capped = [i for i in range(n_opps) if target[i] > cap]
        if capped:
            excess = sum(target[i] - cap for i in capped)
            for i in capped:
                target[i] = cap
            # Redistribute excess to under-capped opps by their weight.
            # Iterate until excess is drained or all opps hit cap (rare).
            uncapped = [i for i in range(n_opps) if target[i] < cap]
            while excess > 0 and uncapped:
                uncap_w = sum(weights[i] for i in uncapped)
                if uncap_w <= 0:
                    break
                added_this_round = 0
                still_uncapped = []
                for i in uncapped:
                    add = max(1, int(round(excess * weights[i] / uncap_w)))
                    headroom = cap - target[i]
                    add = min(add, headroom, excess - added_this_round)
                    if add <= 0:
                        continue
                    target[i] += add
                    added_this_round += add
                    if target[i] < cap:
                        still_uncapped.append(i)
                if added_this_round == 0:
                    break  # safety: avoid infinite loop
                excess -= added_this_round
                uncapped = still_uncapped
            # Last-resort drift fix: if integer rounding left excess > 0,
            # dump remainder on the highest-weight opp with headroom.
            # Preserves PFSP ordering and total_games invariant.
            if excess > 0:
                remaining = [i for i in range(n_opps) if target[i] < cap]
                if remaining:
                    best_idx = max(remaining, key=lambda i: weights[i])
                    target[best_idx] += excess
                else:
                    # Pathological: every opp at cap. Put on highest-weight
                    # capped opp (better than dropping games).
                    target[weights.index(max(weights))] += excess

    # Step 2: distribute workers across opps.
    # S67-ext speed-aware: when a slow opp (mcts-hard, ~30s/game) is in the
    # active pool, swap the greedy cost metric from `games` to `games *
    # per_game_seconds` so extra workers flow toward the slow opp instead of
    # piling on opps that are already fast. When no slow opp is present, this
    # is identical to the legacy games-balanced greedy (each opp converges to
    # workers ~ proportional to game share, i.e., 3 per opp for 30w/10opp).
    per_game_s = [_estimate_per_game_seconds(o) for o in opp_pool_msg]
    has_slow_opp = any(s >= _SLOW_OPP_THRESHOLD_S for s in per_game_s)
    cost = (
        [target[i] * per_game_s[i] for i in range(n_opps)] if has_slow_opp
        else list(target)
    )
    # Per-opp worker cap in speed-aware mode:
    #   - SP (kind='local'): cap at _SP_CAP_IN_SPEED_AWARE (asyncio.gather
    #     concurrency means more workers don't reduce wall).
    #   - MM/FP (kind='external_subprocess'): cap at instance count from
    #     'instance_ports' field. Adding workers beyond instance count
    #     forces multiple workers to share a single subprocess, undoing
    #     the multi-instance fan-out (S67 instances=N design).
    #   - MCTS (kind='external_inprocess'): no cap (in-process, scales
    #     with CPU which we don't model directly).
    #   - In NON-speed-aware mode, no caps apply — legacy 3w/opp behavior.
    def _worker_cap(i: int) -> Optional[int]:
        if not has_slow_opp:
            return None
        kind = opp_pool_msg[i].get("kind", "local")
        if kind == "local":
            return _SP_CAP_IN_SPEED_AWARE
        if kind == "external_subprocess":
            inst_ports = opp_pool_msg[i].get("instance_ports") or []
            return len(inst_ports) if inst_ports else None
        return None  # in-process / unknown: no cap
    caps = [_worker_cap(i) for i in range(n_opps)]

    if n_workers >= n_opps:
        # Pre-fill capped opps to their cap (in speed-aware mode). This
        # ensures MMs always get full instance utilization (one worker per
        # subprocess instance) rather than the greedy under-allocating MM
        # because MCTS's higher wall_cost wins the ratio comparison.
        # Then greedy distributes remaining workers across uncapped opps.
        if has_slow_opp:
            workers_per_opp = [caps[i] if caps[i] is not None else 1
                               for i in range(n_opps)]
            # Defensive: if pre-fill exceeds budget, trim back to 1 each
            # (caps were too aggressive for this n_workers).
            if sum(workers_per_opp) > n_workers:
                workers_per_opp = [1] * n_opps
        else:
            workers_per_opp = [1] * n_opps
        extras = n_workers - sum(workers_per_opp)
        for _ in range(extras):
            # Build candidate list, excluding opps that hit their cap.
            candidates = [
                (cost[i] / workers_per_opp[i], i)
                for i in range(n_opps)
                if caps[i] is None or workers_per_opp[i] < caps[i]
            ]
            if candidates:
                workers_per_opp[max(candidates, key=lambda x: x[0])[1]] += 1
            else:
                # Defensive: all opps capped (shouldn't trigger with
                # reasonable mixes since MCTS has no cap).
                ratios = [cost[i] / workers_per_opp[i] for i in range(n_opps)]
                workers_per_opp[ratios.index(max(ratios))] += 1
        if has_slow_opp:
            # Surface the wall-time projection per opp so iter wall variance
            # is debuggable from the train log alone.
            wall_proj = ", ".join(
                f"{(opp_pool_msg[i].get('key') or 'opp')[:24]}="
                f"{cost[i] / workers_per_opp[i]:.0f}s/{workers_per_opp[i]}w"
                for i in range(n_opps)
            )
            print(
                f"[cis-orch] speed-aware alloc (slow opp >= "
                f"{_SLOW_OPP_THRESHOLD_S:.0f}s/g): {wall_proj}",
                flush=True,
            )
    else:
        sorted_idx = sorted(range(n_opps), key=lambda i: -weights[i])
        top_idx = set(sorted_idx[:n_workers])
        workers_per_opp = [1 if i in top_idx else 0 for i in range(n_opps)]
        skipped = sum(target[i] for i in range(n_opps) if i not in top_idx)
        included_total = sum(target[i] for i in top_idx)
        if skipped > 0 and included_total > 0:
            new_target = list(target)
            for i in top_idx:
                new_target[i] = target[i] + int(round(skipped * target[i] / included_total))
            for i in range(n_opps):
                if i not in top_idx:
                    new_target[i] = 0
            drift = total_games - sum(new_target)
            if drift != 0:
                biggest = max(top_idx, key=lambda j: new_target[j])
                new_target[biggest] += drift
            target = new_target

    # Step 3: per-worker assignments. For each opp, split its games among
    # its assigned workers. Note: instance routing for multi-instance external
    # subprocesses is NOT done here — it's deferred to the dispatch loop
    # (cis_iter_dispatch in this file) which has access to server_pool and
    # can co-locate workers + MM instances on the same battle_server port
    # (F4). Here we just pass through the logical opp dict.
    assignments: List[dict] = []
    for opp_idx in range(n_opps):
        n_w = workers_per_opp[opp_idx]
        if n_w == 0:
            continue
        opp_msg = opp_pool_msg[opp_idx]
        games = target[opp_idx]
        base = games // n_w
        rem = games % n_w
        for w in range(n_w):
            assignments.append({
                "opp": opp_msg,
                "n_games": base + (1 if w < rem else 0),
            })

    # Defensive: pad to n_workers (shouldn't trigger if math is right).
    while len(assignments) < n_workers:
        assignments.append({"opp": opp_pool_msg[0], "n_games": 0})
    return assignments[:n_workers]


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
