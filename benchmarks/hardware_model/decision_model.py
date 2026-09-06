"""Pre-compile Newton unroll decision from executed instruction bytes."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
for entry in (str(REPO), str(BENCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.hardware_model import candidate_selection as selection  # noqa: E402
from benchmarks.hardware_model import footprint_calibration as calibration  # noqa: E402
from benchmarks.hardware_model import fetch_rule_score as fetch  # noqa: E402
from benchmarks.hardware_model import implicit_native_lowering as lowering  # noqa: E402
from benchmarks.hardware_model import instruction_footprint as footprint  # noqa: E402

HARDWARE = json.loads(fetch.NOTES.joinpath(
    "verification/placement_v2_author_20260905/cohort_e1/"
    "kvaerno3_bicgstab_request.json").read_text())["hardware"]
ARCHITECTURE = json.loads(Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "implicit_policy_graph_cpu_e1/architecture.json").read_text())
# Median projected-slot to SASS ratio over 345 calibration rows.
SLOT_FACTOR = 1.417
FULL = ("full",) * 8


def levels_for(algo, newton):
    """Eight loop levels with Krylov rolled on iterative solvers."""
    levels = list(FULL)
    levels[6] = newton
    if algo.endswith("_bicgstab"):
        levels[7] = "count1"
    return tuple(levels)


def step_footprint(system_name, algo, levels, folder):
    compiler = json.loads(calibration.COMPILER.read_text())
    graph = calibration.build_graph(system_name, algo, levels, folder)
    wrapper = footprint.construct_typed_body(graph, compiler)
    row = calibration.summarize(graph, wrapper)
    plan = lowering.make_plan(graph, dict(ARCHITECTURE, gpr_budget=255),
                              compiler, "promote")
    regs = plan["allocation"]["peak_resident"]["R"]
    return row, regs


def executed_kb(row, bodies):
    return fetch.executed_slots(row, bodies) * fetch.WIDTH / 1024 / SLOT_FACTOR


def decide(system_name, algo, block, regimes, folder):
    full_row, regs = step_footprint(
        system_name, algo, levels_for(algo, "full"), folder / "full")
    rolled_row, _ = step_footprint(
        system_name, algo, levels_for(algo, "count1"), folder / "count1")
    geometry = selection.residency(HARDWARE, regs, block, 0, 4, 8192)
    warps = geometry["resident_warps_per_sm"]
    curve = fetch.delivery_curve()
    limit = fetch.capacity(curve, warps)
    caps = [item["source_cap"] for item in full_row["coverage"]["recurrent_loops"]
            if item["group"] == "unroll_newton_exits"]
    cap = sum(caps)
    rows = []
    for regime in regimes:
        bodies = cap if regime == "cap" else regime
        h_full = executed_kb(full_row, bodies)
        h_roll = executed_kb(rolled_row, bodies)
        s_full = fetch.service(curve, h_full, warps)
        s_roll = fetch.service(curve, h_roll, warps)
        rolled = s_full / s_roll > 1.02 or (
            h_full > limit and h_roll > limit and h_roll < h_full * 0.98)
        rows.append(dict(newton_bodies_per_step=bodies, full_kb=round(h_full),
                         rolled_kb=round(h_roll), choice="count1" if rolled
                         else "full"))
    return dict(system=system_name, algo=algo, modeled_registers=regs,
                resident_warps=warps, capacity_kb=limit, newton_cap=cap,
                regimes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system")
    parser.add_argument("algo")
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--regimes", default="1,2,4,cap")
    args = parser.parse_args()
    regimes = [r if r == "cap" else int(r) for r in args.regimes.split(",")]
    with tempfile.TemporaryDirectory() as folder:
        result = decide(args.system, args.algo, args.block, regimes,
                        Path(folder))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
