"""Score the instruction-delivery rule against the post882 unroll bank."""

import argparse
import json
from pathlib import Path

CATALOG = Path(__file__).with_name("INSTRUCTION_DELIVERY_CATALOG.json")
NOTES = Path("C:/local_working_projects/cubie-notes/hardware_unroll_placement")
POST882_AUDIT = NOTES / "recovered/post882_audit_strict.json"
COUNTS = NOTES / "iteration_counts_20260904/iter_counts_fixed.jsonl"
WIDTH = 16
ITERATION_GROUPS = ("unroll_newton_exits", "unroll_krylov_exits")


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
        return above[0][1]
    return rows[-1][1]


def capacity(curve, warps):
    """Largest hot size whose service is within 10% of the smallest."""
    warps_key = min(curve, key=lambda w: abs(w - warps))
    rows = curve[warps_key]
    base = rows[0][1]
    return max(kb for kb, ns in rows if ns <= 1.1 * base)


def measured_counts():
    counts = {}
    for line in COUNTS.read_text().splitlines():
        row = json.loads(line)
        attempted = row["attempted"]["mean"]
        counts[(row["system"], row["algo"])] = dict(
            newton=row["newton"]["mean"] / attempted,
            krylov=row["krylov"]["mean"] / attempted,
        )
    return counts


def observations(audit, policies):
    out = {}
    for config in audit["configs"]:
        key = (config["system"], config["algo"])
        best = {}
        for obs in config.get("observations", []):
            if not obs.get("eligible") or obs["policy"] not in policies:
                continue
            ratio = obs["ratio_to_full"]
            occ = obs["resources"]["occupancy"]["default"]
            label = obs["policy"]
            if label not in best or ratio < best[label]["ratio"]:
                best[label] = dict(ratio=ratio, regs=obs["resources"]["regs"],
                                   sass=obs["resources"]["sass_instructions"],
                                   warps=occ["resident_threads"] // 32)
        out[key] = best
    return out


def executed_slots(row, bodies_per_step):
    """Executed slots per step at the given iteration count per step."""
    scenario = row["scenarios"]["inline|rolled"]
    covered = scenario["covered_slots"]
    cap = scenario["cap_slots"]
    loops = [item for item in row["coverage"]["recurrent_loops"]
             if item["group"] in ITERATION_GROUPS]
    # One body per loop is visited; the cap projection reserves source_cap.
    visited = len(loops)
    reserved = sum(item["source_cap"] for item in loops)
    if not loops or reserved == visited or bodies_per_step is None:
        return cap
    per_body = (cap - covered) / (reserved - visited)
    executed = covered + per_body * max(0.0, bodies_per_step - visited)
    return min(executed, cap)


def slot_factor(rows):
    """Median projected-slots to SASS ratio over resolvable rows."""
    ratios = sorted(
        row["scenarios"]["inline|rolled"]["cap_slots"]
        / row["sass"]["instructions"]
        for row in rows.values() if row["status"] == "ok"
        and row["scenarios"]["inline|rolled"]["cap_slots"]
        < 3 * row["sass"]["instructions"]
    )
    return ratios[len(ratios) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--full", default="u1111111")
    parser.add_argument("--rolled", default="u1111110")
    parser.add_argument("--regime", choices=("measured", "cap"),
                        default="measured")
    parser.add_argument("--lu-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    rows = {}
    for line in Path(args.records).read_text().splitlines():
        row = json.loads(line)
        rows[(row["system"], row["algo"], row["policy"])] = row
    factor = slot_factor(rows)
    print(f"projected-slot factor {factor:.3f} (projected / SASS median)")
    curve = delivery_curve()
    counts = measured_counts()
    audit = json.loads(POST882_AUDIT.read_text())
    measured = observations(audit, {args.full, args.rolled})
    print(f"{'config':34s} {'N/step':>6s} {'Hfull':>6s} {'Hroll':>6s} "
          f"{'warps':>5s} {'pred':>6s} {'meas':>6s} verdict")
    tally = dict(captured=0, MISSED=0, tie=0)
    for (system, algo), best in sorted(measured.items()):
        if args.lu_only and algo.endswith("_bicgstab"):
            continue
        full = rows.get((system, algo, args.full))
        rolled = rows.get((system, algo, args.rolled))
        if (not full or not rolled or full["status"] != "ok"
                or rolled["status"] != "ok"
                or args.full not in best or args.rolled not in best):
            continue
        if best[args.full]["sass"] == best[args.rolled]["sass"]:
            continue
        bodies = None
        if args.regime == "measured":
            count = counts.get((system, algo))
            if count is None:
                continue
            bodies = count["newton"] if count["newton"] else count["krylov"]
        h_full = executed_slots(full, bodies) * WIDTH / 1024 / factor
        h_roll = executed_slots(rolled, bodies) * WIDTH / 1024 / factor
        warps = best[args.full]["warps"]
        s_full = service(curve, h_full, warps)
        s_roll = service(curve, h_roll, best[args.rolled]["warps"])
        predicted = s_full / s_roll
        ratio = best[args.rolled]["ratio"]
        pred_rolled = predicted > 1.0 + args.tolerance
        limit = capacity(curve, warps)
        # Beyond capacity the smaller executed footprint is preferred.
        if (not pred_rolled and h_full > limit and h_roll > limit
                and h_roll < h_full * (1.0 - args.tolerance)):
            pred_rolled = True
        if 0.95 <= ratio <= 1.05:
            verdict = "tie"
        elif pred_rolled == (ratio < 0.95):
            verdict = "captured"
        else:
            verdict = "MISSED"
        tally[verdict] += 1
        print(f"{system + '/' + algo:34s} {bodies if bodies else 0:6.2f} "
              f"{h_full:6.0f} {h_roll:6.0f} {warps:5d} {predicted:6.2f} "
              f"{ratio:6.3f} {verdict}")
    print(f"\ncaptured {tally['captured']} missed {tally['MISSED']} "
          f"measured ties {tally['tie']}")


if __name__ == "__main__":
    main()
