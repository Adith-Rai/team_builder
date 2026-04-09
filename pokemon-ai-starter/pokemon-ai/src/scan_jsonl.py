# src/scan_jsonl.py  (drop-in)
import argparse, glob, json, os, math
from collections import defaultdict, Counter
from typing import List, Any, Tuple

EPS = 1e-8

def _is_nonzero_ctx(ctx) -> bool:
    try:
        for v in ctx:
            x = float(v)
            if math.fabs(x) > EPS:
                return True
    except Exception:
        return False
    return False

def _all_zero(vec) -> bool:
    try:
        for v in vec:
            if abs(float(v)) > EPS:
                return False
        return True
    except Exception:
        return True  # treat malformed as zero

def _any_nonzero(vec) -> bool:
    try:
        for v in vec:
            if abs(float(v)) > EPS:
                return True
        return False
    except Exception:
        return False

def _winner_ok(winner: Any, result: Any) -> Tuple[bool, str]:
    if result not in (0, 1):
        return (False, f"result={result} not in {{0,1}}")
    if winner == "our":
        return (result == 1, f"winner=our but result={result}")
    if winner in ("opp", "tie", "opp_forced_timeout", "opp_forced_unknown", "opp_forced_turncap", "unknown"):
        return (result == 0, f"winner={winner} but result={result}")
    if winner is None:
        return (False, "winner=None on terminal row")
    return (False, f"winner='{winner}' unrecognized")

def pick_latest(files: List[str], n: int) -> List[str]:
    if n <= 0 or n >= len(files):
        return files
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
    return files_sorted[-n:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="file glob, e.g. data/datasets/obs/*.jsonl")
    ap.add_argument("--max", type=int, default=0, help="max rows per file to scan (0=all)")
    ap.add_argument("--latest", type=int, default=0, help="only scan the N most recently modified matching files")
    ap.add_argument("--show-missing-ctx-episodes", type=int, default=0,
                    help="print up to K episode_ids that never had non-zero ctx_extra")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No match: {args.glob}")
        return
    files = pick_latest(files, args.latest)

    grand_rows = grand_eps = grand_steps = schema_issues = 0
    G_ctx_rows_present = G_ctx_rows_nonzero = 0
    G_ctx_opp_last_rows = 0
    G_term_rows = G_term_rows_ok = G_term_rows_bad = 0
    G_nonterm_rows_with_terminal_labels = 0

    # H3 totals
    G_slot_rows_present = 0
    G_move_dims, G_switch_dims = set(), set()
    G_mv_zero_but_legal = G_mv_nz_but_illegal = 0
    G_sw_zero_but_legal = G_sw_nz_but_illegal = 0

    print(f"[scan] {len(files)} file(s) to scan" + (f" (latest {args.latest})" if args.latest else ""))

    for f in files:
        seen = bad = 0
        per_ep = defaultdict(int)
        have_done = Counter()
        first_obs_dim = None

        # ctx
        ctx_rows_present = ctx_rows_nonzero = 0
        ctx_opp_last_rows = 0
        ep_ctx_nonzero = defaultdict(bool)

        # terminals
        term_rows = term_rows_ok = term_rows_bad = 0
        nonterm_rows_with_terminal_labels = 0

        # H3 per-file
        slot_rows_present = 0
        mv_dims, sw_dims = set(), set()
        mv_zero_but_legal = mv_nz_but_illegal = 0
        sw_zero_but_legal = sw_nz_but_illegal = 0

        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if args.max and seen >= args.max:
                        break
                    seen += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        bad += 1
                        continue

                    eid = obj.get("episode_id")
                    t   = obj.get("t")
                    obs = obj.get("obs")
                    done= obj.get("done")
                    if not isinstance(eid, str) or not isinstance(t, int) or not isinstance(obs, list):
                        bad += 1
                        continue
                    if first_obs_dim is None:
                        first_obs_dim = len(obs)

                    per_ep[eid] += 1
                    if done is True:
                        have_done[eid] += 1

                    # ---- ctx checks ----
                    ctx = obj.get("ctx_extra", None)
                    if ctx is not None:
                        ctx_rows_present += 1
                        if _is_nonzero_ctx(ctx):
                            ctx_rows_nonzero += 1
                            ep_ctx_nonzero[eid] = True
                        # Opp-last detection: last 3 dims are kind one-hot [NONE,MOVE,SWITCH]
                        if isinstance(ctx, list) and len(ctx) >= 3:
                            tail3 = ctx[-3:]
                            if _any_nonzero(tail3):
                                ctx_opp_last_rows += 1

                    # ---- mask presence & shape ----
                    legal = obj.get("legal")
                    if not (isinstance(legal, list) and len(legal) == 9):
                        bad += 1
                        continue

                    # ---- H3 slot tensors checks ----
                    mv = obj.get("move_slots")
                    sw = obj.get("switch_slots")
                    if isinstance(mv, list) and len(mv) == 4 and isinstance(mv[0], list) \
                       and isinstance(sw, list) and len(sw) == 5 and isinstance(sw[0], list):
                        slot_rows_present += 1
                        M = len(mv[0]); S = len(sw[0])
                        mv_dims.add(M); sw_dims.add(S)

                        # moves: indices 0..3
                        for i in range(4):
                            v = mv[i] if i < len(mv) else []
                            is_zero = _all_zero(v); is_nz = _any_nonzero(v)
                            if legal[i] == 1:
                                if is_zero: mv_zero_but_legal += 1
                            else:
                                if is_nz:   mv_nz_but_illegal += 1
                        # switches: indices 4..8
                        for j in range(5):
                            v = sw[j] if j < len(sw) else []
                            is_zero = _all_zero(v); is_nz = _any_nonzero(v)
                            if legal[4+j] == 1:
                                if is_zero: sw_zero_but_legal += 1
                            else:
                                if is_nz:   sw_nz_but_illegal += 1

                    # ---- terminal labeling ----
                    winner = obj.get("winner", None)
                    result = obj.get("result", None)
                    if done is True:
                        term_rows += 1
                        ok, _ = _winner_ok(winner, result)
                        if ok: term_rows_ok += 1
                        else:  term_rows_bad += 1
                    else:
                        if (winner is not None) or (result is not None):
                            nonterm_rows_with_terminal_labels += 1

        except Exception as e:
            print(f"[scan] {f}: ERROR {e}")
            continue

        eps = len(per_ep)
        steps = sum(per_ep.values())
        complete_eps = sum(1 for k in per_ep if have_done[k] > 0)

        # roll up totals
        grand_rows += seen
        grand_steps += steps
        grand_eps   += eps
        schema_issues += bad

        G_ctx_rows_present += ctx_rows_present
        G_ctx_rows_nonzero += ctx_rows_nonzero
        G_ctx_opp_last_rows += ctx_opp_last_rows

        G_term_rows        += term_rows
        G_term_rows_ok     += term_rows_ok
        G_term_rows_bad    += term_rows_bad
        G_nonterm_rows_with_terminal_labels += nonterm_rows_with_terminal_labels

        G_slot_rows_present += slot_rows_present
        G_move_dims |= mv_dims
        G_switch_dims |= sw_dims
        G_mv_zero_but_legal += mv_zero_but_legal
        G_mv_nz_but_illegal += mv_nz_but_illegal
        G_sw_zero_but_legal += sw_zero_but_legal
        G_sw_nz_but_illegal += sw_nz_but_illegal

        # Per-file summary
        ctx_pct = (ctx_rows_nonzero / ctx_rows_present * 100.0) if ctx_rows_present else 0.0
        opp_last_pct = (ctx_opp_last_rows / ctx_rows_present * 100.0) if ctx_rows_present else 0.0
        mv_dim_str = ",".join(map(str, sorted(mv_dims))) if mv_dims else "-"
        sw_dim_str = ",".join(map(str, sorted(sw_dims))) if sw_dims else "-"
        print(
            f"[scan] {os.path.basename(f)}: "
            f"rows={seen} eps={eps} complete_eps={complete_eps} "
            f"avg_steps/ep={steps/eps if eps else 0:.2f} obs_dim={first_obs_dim} bad_rows={bad} | "
            f"ctx_rows={ctx_rows_present} nonzero_ctx_rows={ctx_rows_nonzero} ({ctx_pct:.1f}%) "
            f"opp_last_rows={ctx_opp_last_rows} ({opp_last_pct:.1f}%) | "
            f"slot_rows={slot_rows_present} M={mv_dim_str} S={sw_dim_str} | "
            f"anoms: mv_zero&legal={mv_zero_but_legal} mv_nz&illegal={mv_nz_but_illegal} "
            f"sw_zero&legal={sw_zero_but_legal} sw_nz&illegal={sw_nz_but_illegal} | "
            f"nonterm_rows_with_terminal_labels={nonterm_rows_with_terminal_labels}"
        )
        if len(mv_dims) > 1 or len(sw_dims) > 1:
            print(f"[scan][warn] {os.path.basename(f)}: slot dim mismatch within file "
                  f"(M={mv_dim_str}, S={sw_dim_str})")

        # Optionally list episodes that never had non-zero ctx
        if args.show_missing_ctx_episodes > 0 and eps > 0:
            missing = [eid for eid in per_ep.keys() if not ep_ctx_nonzero[eid]]
            if missing:
                sample = missing[:args.show_missing_ctx_episodes]
                tail = f" (showing {len(sample)}/{len(missing)})" if len(missing) > len(sample) else ""
                print(f"[scan] episodes with NO non-zero ctx in {os.path.basename(f)}: {len(missing)}{tail}")
                for eid in sample:
                    print(f"  - {eid}")

    # Grand totals
    grand_avg = (grand_steps/grand_eps) if grand_eps else 0.0
    grand_ctx_pct = (G_ctx_rows_nonzero / G_ctx_rows_present * 100.0) if G_ctx_rows_present else 0.0
    grand_opp_last_pct = (G_ctx_opp_last_rows / G_ctx_rows_present * 100.0) if G_ctx_rows_present else 0.0
    mv_dim_str = ",".join(map(str, sorted(G_move_dims))) if G_move_dims else "-"
    sw_dim_str = ",".join(map(str, sorted(G_switch_dims))) if G_switch_dims else "-"

    print(f"[scan] TOTAL: files={len(files)} rows={grand_rows} episodes={grand_eps} avg_steps/ep={grand_avg:.2f} bad_rows={schema_issues}")
    print(f"[scan] TOTAL: ctx_rows={G_ctx_rows_present} nonzero_ctx_rows={G_ctx_rows_nonzero} ({grand_ctx_pct:.1f}%) "
          f"opp_last_rows={G_ctx_opp_last_rows} ({grand_opp_last_pct:.1f}%)")
    print(f"[scan] TOTAL: slot_rows={G_slot_rows_present} M={mv_dim_str} S={sw_dim_str}")
    print(f"[scan] TOTAL: anoms: mv_zero&legal={G_mv_zero_but_legal} mv_nz&illegal={G_mv_nz_but_illegal} "
          f"sw_zero&legal={G_sw_zero_but_legal} sw_nz&illegal={G_sw_nz_but_illegal}")

if __name__ == "__main__":
    main()
