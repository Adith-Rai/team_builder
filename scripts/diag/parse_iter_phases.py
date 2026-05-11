#!/usr/bin/env python3
"""Parse [FLOW ...] + [HH:MM:SS] Iter N: lines from a train_rl log → per-phase
wall-time table. Reads from stdin. Usage:
    grep -E '^\\[FLOW |^\\[..:..:..\\] Iter [0-9]+:' /tmp/run.log | python3 parse_iter_phases.py
"""
import sys
import re

lines = sys.stdin.read().split("\n")
iters = []
cur = None
for line in lines:
    m = re.search(r"\[FLOW (\d\d:\d\d:\d\d) \+\s*([\d.]+)s\]\s*(.*)", line)
    n = re.search(r"^\[(\d\d:\d\d:\d\d)\]\s*Iter (\d+):.*collect=(\d+)s, update=(\d+)s", line)
    if m:
        ts, secs, msg = m.groups()
        secs = float(secs)
        if "iter start" in msg:
            if cur and "iter_n" in cur:
                iters.append(cur)
            cur = {"start_ts": ts, "phases": {"iter_start": secs}}
        elif cur is not None:
            for key, pat in [
                ("cis_start", "starting CIS"),
                ("collect_done", "cis collect done"),
                ("eps_built", "PPO episodes built"),
                ("update_start", "starting PPO update"),
            ]:
                if pat in msg:
                    cur["phases"][key] = secs
    elif n and cur is not None:
        ts, idx, c, u = n.groups()
        cur["iter_n"] = int(idx)
        cur["end_ts"] = ts
        cur["c_rep"] = int(c)
        cur["u_rep"] = int(u)
        iters.append(cur)
        cur = None

hdr = "iter | collect_s | eps_build | update_s | total_s | end_ts"
print(hdr)
print("-" * len(hdr))
for it in iters:
    if "iter_n" not in it:
        continue
    p = it["phases"]
    cs = it["c_rep"]
    us = it["u_rep"]
    if "collect_done" in p and "update_start" in p:
        eb = round(p["update_start"] - p["collect_done"], 1)
    else:
        eb = "-"
    total = cs + (eb if isinstance(eb, (int, float)) else 0) + us
    eb_s = f"{eb:>9}" if isinstance(eb, str) else f"{eb:>9.1f}"
    print(f"{it['iter_n']:>4} | {cs:>9} |{eb_s} | {us:>8} | {total:>7} | {it['end_ts']:>8}")
