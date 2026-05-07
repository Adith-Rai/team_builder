"""Single source of truth for autocast precision (fp16 vs bf16 vs fp32).

Why this module exists: prior to Session 51, every autocast site in the codebase
hardcoded `torch.amp.autocast("cuda", enabled=fp16)` which defaults to
`dtype=torch.float16`. Adding bf16 support (no GradScaler, fp32 dynamic range,
same Tensor Core throughput on Ampere) without touching every autocast call
signature is achieved by reading a module-level setting in one place.

Set the global dtype ONCE at process start (in train_rl.py main, or via cmd
msg on worker side for mp paths). Then every `with autocast_ctx(fp16=...)`
call picks up the global setting transparently. The `fp16` kwarg remains as a
legacy fallback when the global isn't set (e.g., eval scripts that haven't
been updated yet).

Behavior:
  set_amp_dtype(torch.bfloat16) -> all autocast sites use bf16
  set_amp_dtype(torch.float16)  -> all autocast sites use fp16
  set_amp_dtype(None)           -> autocast follows legacy fp16 bool arg

Multi-process notes (mp_disk_collect.py, etc.):
  Each worker process is a separate Python interpreter, so the global must be
  set on the worker side too. The `cmd` dict passed via mp.Pipe carries an
  `amp_dtype: str` field ("fp16" / "bf16" / "fp32"); worker translates and
  calls `set_amp_dtype()` on receipt before any forward call.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Union

import torch


# Module-level setting. None means "follow legacy fp16 bool arg".
_AMP_DTYPE: Optional[torch.dtype] = None


def set_amp_dtype(dtype: Optional[torch.dtype]) -> None:
    """Set the global amp dtype. Call once at process start.

    Pass None to revert to legacy behavior (each autocast site uses its own
    fp16 bool arg). Pass torch.float16 / torch.bfloat16 to force a dtype
    everywhere.
    """
    global _AMP_DTYPE
    _AMP_DTYPE = dtype


def get_amp_dtype() -> Optional[torch.dtype]:
    return _AMP_DTYPE


def parse_amp_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Convert a string ('fp16'|'bf16'|'fp32'|None) to torch.dtype or None.

    Useful when shipping the dtype across processes via JSON/pickle dicts."""
    if name is None or name == "fp32":
        return None
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unknown amp dtype name: {name!r}")


def amp_dtype_name(dtype: Optional[torch.dtype]) -> str:
    """Inverse of parse_amp_dtype - serialize a torch.dtype for IPC."""
    if dtype is None:
        return "fp32"
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return str(dtype)


def autocast_ctx(fp16: bool = False) -> contextlib.AbstractContextManager:
    """Context manager for autocast that respects the global amp dtype.

    Args:
        fp16: legacy fallback. Used only when no global amp dtype is set
              (i.e., set_amp_dtype was never called or was passed None).
              When True with no global, autocast in fp16; when False, no
              autocast.

    Usage replaces the prior pattern:
        # OLD: with torch.amp.autocast("cuda", enabled=self.fp16):
        # NEW: with autocast_ctx(self.fp16):
    """
    if _AMP_DTYPE is not None:
        return torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE)
    if fp16:
        return torch.amp.autocast("cuda", enabled=True, dtype=torch.float16)
    return torch.amp.autocast("cuda", enabled=False)
