"""Aggregate viztracer chrome-trace JSON output into a human-readable report.

Usage:
  python scripts/bench/analyze_viztracer.py <profile_*.json> [<profile_*.json> ...]

For each JSON, reports:
  - process wall time
  - top N functions by inclusive self-time
  - top N functions by call count
  - rough breakdown of where time goes by module/file

Chrome-trace JSON format:
  {"traceEvents": [
    {"ph": "B", "name": "func_name", "ts": 1234, "tid": ..., "pid": ...},
    {"ph": "E", "name": "func_name", "ts": 1567, ...},
    ...
  ]}
  Timestamps in microseconds.

  Duration events have ph="X" with "dur" field directly.

This script handles both X-events (with dur) and B/E pairs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Optional


def load_trace(path: str) -> list:
    # Viztracer JSON may include non-ASCII bytes in name fields; force UTF-8
    # with replace so we don't bail on minor encoding hiccups.
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'traceEvents' in data:
        return data['traceEvents']
    if isinstance(data, list):
        return data
    raise ValueError(f'Unexpected JSON shape in {path}')


def aggregate(events: list) -> tuple[dict, dict, int, int]:
    """Returns (per_func_total_us, per_func_count, total_us, n_events_used)."""
    totals: dict = defaultdict(int)
    counts: dict = defaultdict(int)
    # For B/E pairs we track open stacks per (pid, tid)
    open_stacks: dict = defaultdict(list)

    earliest_ts: Optional[int] = None
    latest_end: Optional[int] = None
    used = 0

    for ev in events:
        ph = ev.get('ph')
        name = ev.get('name', '?')
        ts = ev.get('ts')
        if ts is None:
            continue
        if earliest_ts is None or ts < earliest_ts:
            earliest_ts = ts

        if ph == 'X':
            dur = ev.get('dur', 0)
            totals[name] += dur
            counts[name] += 1
            end = ts + dur
            if latest_end is None or end > latest_end:
                latest_end = end
            used += 1
        elif ph == 'B':
            key = (ev.get('pid'), ev.get('tid'))
            open_stacks[key].append((name, ts))
        elif ph == 'E':
            key = (ev.get('pid'), ev.get('tid'))
            stack = open_stacks[key]
            if not stack:
                continue
            n, start_ts = stack.pop()
            if n != name:
                # Mismatched event; skip
                continue
            dur = ts - start_ts
            totals[name] += dur
            counts[name] += 1
            if latest_end is None or ts > latest_end:
                latest_end = ts
            used += 1

    total_us = (latest_end - earliest_ts) if (earliest_ts is not None and latest_end is not None) else 0
    return dict(totals), dict(counts), total_us, used


def fmt_us(us: int) -> str:
    if us < 1000:
        return f'{us}us'
    if us < 1000_000:
        return f'{us/1000:.1f}ms'
    return f'{us/1_000_000:.2f}s'


def report(path: str, top_n: int = 30) -> None:
    print('=' * 80)
    print(f'FILE: {path}')
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f'Size: {size_mb:.1f} MB')

    events = load_trace(path)
    print(f'Events: {len(events)}')

    totals, counts, total_us, used = aggregate(events)
    print(f'Trace span: {fmt_us(total_us)} ({used} timed events used)')
    print()

    sorted_total = sorted(totals.items(), key=lambda kv: -kv[1])
    print(f'TOP {top_n} by inclusive total time:')
    print(f'  {"function":<55}  {"total":>10}  {"calls":>8}  {"avg":>10}  {"%wall":>7}')
    cumulative = 0
    for name, t in sorted_total[:top_n]:
        c = counts.get(name, 0)
        avg = t / c if c else 0
        pct = (t / total_us * 100) if total_us else 0
        cumulative += pct
        n_short = name if len(name) <= 53 else name[:50] + '...'
        print(f'  {n_short:<55}  {fmt_us(t):>10}  {c:>8}  {fmt_us(int(avg)):>10}  {pct:>6.2f}%')
    print(f'  TOP {top_n} cumulative % of wall: {cumulative:.1f}%')
    print()

    sorted_count = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f'TOP {min(top_n, len(sorted_count))} by call count:')
    print(f'  {"function":<55}  {"calls":>8}  {"total":>10}  {"avg":>10}')
    for name, c in sorted_count[:top_n]:
        t = totals.get(name, 0)
        avg = t / c if c else 0
        n_short = name if len(name) <= 53 else name[:50] + '...'
        print(f'  {n_short:<55}  {c:>8}  {fmt_us(t):>10}  {fmt_us(int(avg)):>10}')
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        report(p, top_n=30)


if __name__ == '__main__':
    main()
