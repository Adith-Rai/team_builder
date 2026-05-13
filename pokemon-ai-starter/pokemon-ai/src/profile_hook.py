"""Diagnostic profiling hooks — env-var-gated, zero impact when PROFILE_MODE unset.

Usage:
  At each process entrypoint, call:
      from profile_hook import maybe_start_viztracer
      maybe_start_viztracer('worker')   # or 'cis' or 'main'

  Output: /tmp/profile_<name>_<pid>.json (viztracer chrome-trace format)
  View locally with: vizviewer /tmp/profile_main_*.json
  Or open in chrome://tracing or https://ui.perfetto.dev

  Optional manual timers:
      from profile_hook import Timer, timing_dump
      with Timer('phase_name'):
          ...
      atexit.register(timing_dump)   # prints aggregated stats at process exit

Behaviour when PROFILE_MODE != '1':
  - maybe_start_viztracer: returns None, does nothing
  - Timer: no-op context manager
  - timing_dump: no-op

This module imports viztracer LAZILY (only when PROFILE_MODE is on) so the
import itself is free in normal runs.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Optional


_PROFILE_ON = os.environ.get('PROFILE_MODE') == '1'
_TIMINGS: dict = defaultdict(lambda: {'total': 0.0, 'count': 0, 'max': 0.0})


def is_on() -> bool:
    return _PROFILE_ON


def maybe_start_viztracer(name: str, max_entries: int = 1_000_000) -> Optional[object]:
    """Start a per-process viztracer if PROFILE_MODE=1.

    Returns the tracer (or None if disabled / failed). Auto-saves on:
      - SIGUSR1 (synchronous, exits with code 0)
      - process exit (atexit)
    """
    if not _PROFILE_ON:
        return None
    try:
        from viztracer import VizTracer  # type: ignore
    except ImportError:
        print(f'[PROFILE] viztracer not installed; skipping ({name})',
              flush=True, file=sys.stderr)
        return None

    pid = os.getpid()
    output = f'/tmp/profile_{name}_{pid}.json'
    tracer = VizTracer(
        output_file=output,
        max_stack_depth=18,
        tracer_entries=max_entries,
        ignore_c_function=True,
        ignore_frozen=True,
        log_func_args=False,
        log_print=False,
    )
    try:
        tracer.start()
    except Exception as e:
        print(f'[PROFILE] viztracer.start() failed for {name}: {e}',
              flush=True, file=sys.stderr)
        return None

    # Double-save protection — SIGUSR1 + atexit both want to save
    saved = {'done': False}

    def _do_save(reason: str) -> None:
        if saved['done']:
            return
        saved['done'] = True
        try:
            tracer.stop()
            tracer.save()
            print(f'[PROFILE] saved viztracer for {name} via {reason} -> {output}',
                  flush=True, file=sys.stderr)
        except Exception as e:
            print(f'[PROFILE] viztracer save failed for {name} via {reason}: {e}',
                  flush=True, file=sys.stderr)

    def _sigusr1_handler(signum, frame):
        _do_save('SIGUSR1')
        # Exit cleanly so any other atexit/cleanup fires too
        os._exit(0)

    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
    except (ValueError, OSError) as e:
        # Some thread contexts (non-main) reject signal.signal — ignore
        print(f'[PROFILE] could not install SIGUSR1 handler for {name}: {e}',
              flush=True, file=sys.stderr)

    atexit.register(lambda: _do_save('atexit'))

    print(f'[PROFILE] viztracer started for {name} (pid={pid}, out={output}, '
          f'sigusr1=on)',
          flush=True, file=sys.stderr)
    return tracer


@contextmanager
def Timer(name: str) -> Iterator[None]:
    """Aggregating timer context manager. Sums total time + counts hits.

    No-op when PROFILE_MODE != '1'.
    """
    if not _PROFILE_ON:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        entry = _TIMINGS[name]
        entry['total'] += dt
        entry['count'] += 1
        if dt > entry['max']:
            entry['max'] = dt


def timing_dump(tag: str = '') -> None:
    """Print aggregated timing stats. Call at process exit."""
    if not _PROFILE_ON:
        return
    if not _TIMINGS:
        return
    pid = os.getpid()
    print(f'\n[PROFILE-TIMINGS pid={pid} {tag}]', flush=True, file=sys.stderr)
    items = sorted(_TIMINGS.items(), key=lambda kv: -kv[1]['total'])
    name_w = max(len(k) for k in _TIMINGS)
    print(f'  {"name":<{name_w}}  {"total_s":>10}  {"count":>8}  {"avg_ms":>10}  {"max_ms":>10}',
          flush=True, file=sys.stderr)
    for name, s in items:
        avg_ms = (s['total'] / s['count']) * 1000 if s['count'] else 0.0
        print(f'  {name:<{name_w}}  {s["total"]:>10.3f}  {s["count"]:>8}  '
              f'{avg_ms:>10.3f}  {s["max"] * 1000:>10.3f}',
              flush=True, file=sys.stderr)


def register_timing_dump(tag: str = '') -> None:
    """Convenience: register timing_dump at exit."""
    if _PROFILE_ON:
        atexit.register(lambda: timing_dump(tag))
