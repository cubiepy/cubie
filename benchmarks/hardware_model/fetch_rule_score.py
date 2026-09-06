"""Score the instruction-delivery rule against the measured unroll banks.

For every audited configuration the projected per-step instruction bytes
of the all-full and iteration-rolled policies are compared with the
measured instruction-delivery curve, and the predicted winner is scored
against the eligible measured ratio.
"""

import argparse
import json
from pathlib import Path

CATALOG = Path(__file__).with_name("INSTRUCTION_DELIVERY_CATALOG.json")
POST882_AUDIT = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "recovered/post882_audit_strict.json"
)
SPLIT_AUDIT = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "recovered/split_flags_audit_strict.json"
)
WIDTH = 16


def delivery_curve():
    catalog = json.loads(CATALOG.read_text())
    points = {}
    for row in catalog["points"]:
        if row["unroll_flag"] != "True" or row["hot_kb"] >= 200:
            continue
        points.setdefault(row["warps_per_sm"], []).append(
            (row["hot_kb"], row["ns_per_warp_instruction"]))
    for warps in points:
        points[warps].sort()
    return points


def service(curve, hot_kb, warps):
    """Nearest measured point at or above hot_kb for the closest warps."""
    warps_key = min(curve, key=lambda w: abs(w - warps))
    rows = curve[warps_key]
    above = [r for r in rows if r[0] >= hot_kb]
    if above:
        return above[0][1], above[0][0], warps_key
    return rows[-1][1], rows[-1][0], warps_key


def observations(audit, policies):
    out = {}
    for config in audit["configs"]:
        key = (config["system"], config["algo"])
        best = {}
        for obs in config.get("observations", []):
            if not obs.get("eligible"):
                continue
            label = obs["policy"]
            if label not in policies:
                continue
            ratio = obs["ratio_to_full"]
            regs = obs["resources"]["regs"]
            occ = obs["resources"]["occupancy"]["default"]
            if label not in best or ratio < best[label]["ratio"]:
                best[label] = dict(ratio=ratio, regs=regs,
                                   sass=obs["resources"]["sass_instructions"],
                                   warps=occ["resident_threads"] // 32)
        out[key] = best
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--full", default="u1111111")
    parser.add_argument("--rolled", default="u1111110")
    parser.add_argument("--newton-bodies", type=int, default=0,
                        help="executed Newton bodies per stage; 0 = cap")
    args = parser.parse_args()
    rows = {}
    for line in Path(args.records).read_text().splitlines():
        row = json.loads(line)
        rows[(row["system"], row["algo"], row["policy"])] = row
    curve = delivery_curve()
    audit = json.loads(POST882_AUDIT.read_text())
    measured = observations(audit, {args.full, args.rolled})
    print(f"{'config':36s} {'Hfull':>6s} {'Hroll':>6s} {'warps':>5s} "
          f"{'sFull':>6s} {'sRoll':>6s} {'pred':>6s} {'meas':>6s} verdict")
    captured = missed = ties = 0
    for (system, algo), best in sorted(measured.items()):
        full = rows.get((system, algo, args.full))
        rolled = rows.get((system, algo, args.rolled))
        if (not full or not rolled or full["status"] != "ok"
                or rolled["status"] != "ok"
                or args.full not in best or args.rolled not in best):
            continue
        if best[args.full]["sass"] == best[args.rolled]["sass"]:
            continue
        sc_full = full["scenarios"]["inline|rolled"]
        sc_roll = rolled["scenarios"]["inline|rolled"]
        h_full = sc_full["cap_slots"] * WIDTH / 1024
        h_roll = sc_roll["cap_slots"] * WIDTH / 1024
        if args.newton_bodies:
            body = (sc_full["cap_slots"] - sc_full["covered_slots"])
            loops = [item for item in full["coverage"]["recurrent_loops"]
                     if item["group"] == "unroll_newton_exits"]
            cap = max((item["source_cap"] for item in loops), default=1)
            if cap > 1:
                h_full = (sc_full["covered_slots"]
                          + body * (min(args.newton_bodies, cap) - 1)
                          / (cap - 1)) * WIDTH / 1024
        warps = best[args.full]["warps"]
        s_full, _, _ = service(curve, h_full, warps)
        s_roll, _, _ = service(curve, h_roll, best[args.rolled]["warps"])
        predicted = s_full / s_roll
        measured_ratio = best[args.rolled]["ratio"]
        pred_rolled_wins = predicted > 1.0
        meas_rolled_wins = measured_ratio < 0.95
        meas_tie = 0.95 <= measured_ratio <= 1.05
        if meas_tie:
            verdict = "tie"
            ties += 1
        elif pred_rolled_wins == meas_rolled_wins:
            verdict = "captured"
            captured += 1
        else:
            verdict = "MISSED"
            missed += 1
        print(f"{system + '/' + algo:36s} {h_full:6.0f} {h_roll:6.0f} "
              f"{warps:5d} {s_full:6.2f} {s_roll:6.2f} {predicted:6.2f} "
              f"{measured_ratio:6.3f} {verdict}")
    print(f"\ncaptured {captured} missed {missed} measured ties {ties}")


if __name__ == "__main__":
    main()
