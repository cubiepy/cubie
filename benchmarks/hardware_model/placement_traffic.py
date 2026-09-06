"""Source memory demand per single-buffer placement for bank configs."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
for entry in (str(REPO), str(BENCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import placement_landscape as landscape  # noqa: E402
from benchmarks.hardware_model import placement_decision as decision  # noqa: E402
from benchmarks.hardware_model import placement_source as source  # noqa: E402
from benchmarks.hardware_model.candidate_selection import canonical  # noqa: E402
from placement_score import partition  # noqa: E402

TEMPLATE = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "verification/placement_v2_author_20260905/cohort_e1/"
    "kvaerno3_bicgstab_request.json"
)
KEEP = ("l1_hits", "l1_misses", "l1_read_requests", "l1_write_requests",
        "l2_read_requests", "l2_write_requests", "l2_misses",
        "local_read_sectors", "local_write_sectors",
        "local_read_warp_instructions", "local_write_warp_instructions",
        "shared_read_warp_instructions", "shared_write_warp_instructions",
        "shared_bank_wavefronts", "dram_read_sectors", "dram_write_sectors")


def inner_of(algo):
    return "bicgstab" if algo.endswith("_bicgstab") else "lu"


def registry_targets(system_name, algo, folder):
    """Owner-qualified relocatable buffers registered at positive size."""
    request = dict(system=system_name, algo=algo, linear_solver=inner_of(algo),
                   newton_bodies=1, krylov_bodies=1, targets=[])
    tableau = algo.partition("_bicgstab")[0]
    if tableau in landscape.NEWTON_TABLEAUS and tableau.startswith("kvaerno"):
        request["branch_choices"] = {"generic_dirk.py:744": False}
    from cubie import Solver
    from cubie.cache_root import get_cache_root_override, set_cache_root
    from benchmarks.hardware_model import implicit_source_graph as graphs
    from benchmarks.hardware_model import implicit_workload as workload

    previous = get_cache_root_override()
    set_cache_root(folder / "probe")
    kwargs = landscape.solver_kwargs(system_name, algo)
    solver = Solver(landscape.SYSTEMS[system_name]["build"](), **kwargs)
    try:
        descriptor = workload.describe_implicit_workload(solver)
        regimes = graphs.uniform_regime(descriptor, 1, 1)
        graph = graphs.describe_implicit_source(
            solver, regimes, request.get("branch_choices", {}))
    finally:
        solver.close()
        set_cache_root(previous)
    targets = []
    for item in graph["registry"]:
        name = item["name"]
        if (item["bytes"] > 0 and name in landscape.BUFFERS
                and item["declared_location"] == "local"):
            targets.append(dict(owner=item["owner"], name=name,
                                setting=landscape.setting_name(name)))
    return request, targets


def demand_rows(request, target, folder, template, blocks_hint):
    graphs = source.construct(dict(request, targets=[target]), folder)
    name = target["owner"] + ":" + target["name"]
    # Each arm uses the partition the driver selects for its dynamic bytes.
    stride = json.loads(Path(graphs[canonical({name: "shared"})]["path"])
                        .read_text())["candidate_construction"][
                            "shared_stride_bytes"]
    block = template["block_threads"]
    carveouts = {"local": partition(blocks_hint, 4),
                 "shared": partition(blocks_hint, stride * block)}
    rows = []
    for space, carveout in sorted(carveouts.items()):
        payload = dict(template)
        payload["named_buffers"] = [name]
        payload["architecture"] = dict(template["architecture"],
                                       gpr_budget=255)
        payload["source_graphs"] = graphs
        payload["cache_scenario"] = dict(template["cache_scenario"],
                                         shared_carveout_bytes=carveout)
        result = decision.enumerate_placements(payload)
        for arm in result["placements"]:
            if arm["placement_identity"][name] != space:
                continue
            demand = arm["memory_demand"]
            rows.append(dict(
                buffer=target["name"], owner=target["owner"],
                space=arm["placement_identity"][payload["named_buffers"][0]],
                materialization=arm["materialization_scenario"],
                carveout=carveout,
                regs=arm["native_plan"]["allocation"]["peak_resident"]["R"],
                frame=arm["native_plan"]["allocation"]["local_frame_bytes"],
                stride=arm["shared_stride_bytes"],
                legal=arm["geometry"]["legal"],
                resident_warps=arm["geometry"].get("resident_warps_per_sm"),
                support=arm["allocation_regime_support"].get("status"),
                waves=None if demand is None else demand["waves"],
                counts=None if demand is None else {
                    k: demand["counts"].get(k, 0) for k in KEEP},
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    template = json.loads(TEMPLATE.read_text())
    records = out / "records.jsonl"
    done = set()
    if records.exists():
        for line in records.read_text().splitlines():
            row = json.loads(line)
            done.add((row["system"], row["algo"], row["buffer"]))
    for item in args.configs.split(","):
        system_name, algo = item.split("/")
        base = out / "graphs" / f"{system_name}__{algo}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            request, targets = registry_targets(system_name, algo, base)
        except Exception:
            print(f"{item} registry failed", flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        for target in targets:
            if (system_name, algo, target["name"]) in done:
                continue
            started = time.perf_counter()
            row = dict(system=system_name, algo=algo, buffer=target["name"],
                       owner=target["owner"])
            try:
                row["arms"] = demand_rows(
                    request, target, base / target["name"], template,
                    args.blocks)
                row["status"] = "ok"
            except Exception:
                row["status"] = "error"
                row["error"] = traceback.format_exc()
            row["seconds"] = time.perf_counter() - started
            with open(records, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"{item:34s} {target['name']:26s} {row['status']:5s} "
                  f"{row['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
