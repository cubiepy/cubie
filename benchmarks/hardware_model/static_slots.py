"""Instruction slots and spill events per runtime region of a policy graph.

One graph per configuration at the full policy (Krylov count 1 on
BiCGSTAB), allocated at 255 registers; each region is (role, body, phase).

```powershell
python benchmarks/hardware_model/static_slots.py --out <dir> --configs chain32/radau_iia_5
```
"""

import argparse
from collections import Counter, defaultdict
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

from benchmarks.hardware_model import footprint_calibration as calibration  # noqa: E402
from benchmarks.hardware_model import implicit_policy_graph as policy  # noqa: E402

ARCHITECTURE = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "implicit_policy_graph_cpu_e1/architecture.json"
)


def region_of(node):
    regions = set()
    for context in node["source_contexts"]:
        region = context.get("runtime_region") or {}
        regions.add((region.get("role"), region.get("body_index"),
                     region.get("phase")))
    if len(regions) != 1:
        return ("ambiguous", None, None)
    return regions.pop()


def summarize(plan):
    nodes = plan["lowering"]["nodes"]
    regions = [region_of(node) for node in nodes]
    slots = Counter()
    spills = Counter()
    reloads = Counter()
    for event in plan["allocation"]["events"]:
        position = event.get("node", event["source_position"])
        region = regions[position]
        if event.get("opcode"):
            slots[region] += 1
        if event["kind"] == "spill":
            spills[region] += 1
        elif event["kind"] == "reload":
            reloads[region] += 1
    defined = {}
    for index, node in enumerate(nodes):
        for value in node["outputs"]:
            defined[value] = regions[index]
    live_in = defaultdict(set)
    for index, node in enumerate(nodes):
        region = regions[index]
        for value in node["inputs"]:
            origin = defined.get(value)
            if origin is not None and origin != region:
                live_in[region].add(value)
    keys = sorted(set(slots) | set(spills) | set(reloads),
                  key=lambda k: (str(k[0]), -1 if k[1] is None else k[1],
                                 str(k[2])))
    return [dict(role=key[0], body=key[1], phase=key[2], slots=slots[key],
                 spills=spills[key], reloads=reloads[key],
                 live_in=len(live_in[key])) for key in keys]


def levels_for(algo):
    levels = ["full"] * 8
    if algo.endswith("_bicgstab"):
        levels[7] = "count1"
    return tuple(levels)


def build(system_name, algo, folder):
    """Graph, plan and region summary for one configuration."""
    architecture = dict(json.loads(ARCHITECTURE.read_text()), gpr_budget=255)
    compiler = json.loads(calibration.COMPILER.read_text())
    levels = levels_for(algo)
    row = dict(system=system_name, algo=algo, levels=list(levels))
    started = time.perf_counter()
    graph = calibration.build_graph(system_name, algo, levels, folder)
    row["graph_nodes"] = len(graph["nodes"])
    plan = policy.special_typed_plan(graph, architecture, compiler, "promote")
    row["registers"] = plan["allocation"]["peak_resident"]["R"]
    row["local_frame_bytes"] = plan["allocation"]["local_frame_bytes"]
    row["regions"] = summarize(plan)
    row["loops"] = [dict(group=item["group"],
                         visits=len(item["executed_instances"]),
                         cap=item["structure"]["fixed_trip_count"])
                    for item in graph["policy_loops"]
                    if item["kind"] == "recurrent_execution_trace"]
    row["seconds"] = time.perf_counter() - started
    return row


def load(records):
    rows = {}
    if Path(records).exists():
        for line in Path(records).read_text().splitlines():
            row = json.loads(line)
            rows[(row["system"], row["algo"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = out / "records.jsonl"
    done = load(records)
    for item in args.configs.split(","):
        system_name, algo = item.split("/")
        if (system_name, algo) in done:
            continue
        try:
            row = build(system_name, algo,
                        out / "build" / f"{system_name}__{algo}")
            row["status"] = "ok"
        except Exception:
            row = dict(system=system_name, algo=algo, status="error",
                       error=traceback.format_exc())
        with open(records, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{item:36s} {row['status']}", flush=True)


if __name__ == "__main__":
    main()
