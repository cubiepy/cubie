"""Compare source-projected instruction slots with post882 SASS counts.

For each historical (system, algorithm, policy) compile row the source
policy graph is rebuilt, lowered to typed opcodes without allocation, and
its slot forecast is written beside the recorded SASS totals and loops.
"""

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

from cubie import Solver  # noqa: E402
from cubie.cache_root import get_cache_root_override, set_cache_root  # noqa: E402

import placement_landscape as landscape  # noqa: E402
from benchmarks.hardware_model import implicit_policy_graph as policy  # noqa: E402
from benchmarks.hardware_model import implicit_workload as workload  # noqa: E402
from benchmarks.hardware_model import instruction_footprint as footprint  # noqa: E402

POST882 = Path("C:/local_working_projects/cubie-notes/unroll_landscape/post882")
COMPILER = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "implicit_policy_graph_cpu_e1/compiler.json"
)
LEVEL_NAMES = {"1": "full", "0": "count1", "2": "count2", "4": "count4",
               "n": "false"}
SEVEN_GROUPS = ("unroll_stage", "unroll_step_element", "unroll_accumulator",
                "unroll_solver_element", "unroll_norms", "unroll_other_small",
                "unroll_converged_exits")
DEFAULT_POLICIES = ("u1111111", "u1111110", "u1111101", "u1111100",
                    "u0111111", "u1011111", "u1101111", "u1110111",
                    "u1111011")


def eight_levels(label):
    """Map a seven-group post882 label onto the eight current groups."""
    if label == "libnvvm":
        return ("false",) * 8
    levels = [LEVEL_NAMES[c] for c in label[1:]]
    return tuple(levels[:6] + [levels[6], levels[6]])


def inner_of(algo):
    tableau = algo.partition("_bicgstab")[0]
    if tableau in landscape.EXPLICIT_TABLEAUS:
        return None
    return "bicgstab" if algo.endswith("_bicgstab") else "lu"


def build_graph(system_name, algo, levels, folder):
    folder.mkdir(parents=True, exist_ok=True)
    previous = get_cache_root_override()
    set_cache_root(folder / "codegen")
    kwargs = landscape.solver_kwargs(system_name, algo)
    kwargs["unroll"] = policy.policy_flags(levels)
    system = landscape.SYSTEMS[system_name]["build"]()
    solver = Solver(system, **kwargs)
    try:
        constants = landscape.SYSTEMS[system_name].get("constants")
        if constants:
            solver.update(constants)
        inner = inner_of(algo)
        if inner is None:
            declared, branches = {}, {}
            fsal = dict(first_step=True, all_lanes_accepted=True)
        else:
            descriptor = workload.describe_implicit_workload(solver)
            from benchmarks.hardware_model import implicit_source_graph
            declared = implicit_source_graph.uniform_regime(descriptor, 1, 1)
            branches = ({"generic_dirk.py:744": False}
                        if descriptor["family"] == "DIRK" else {})
            fsal = None
        graph = policy.describe_policy_source(
            solver, declared, policy.policy_record(levels, kwargs["unroll"]),
            branches, fsal_state=fsal,
        )
        return graph
    finally:
        solver.close()
        set_cache_root(previous)


def summarize(graph, wrapper):
    forecast = footprint.forecast(graph, wrapper)
    out = {}
    for scenario in forecast["scenarios"]:
        key = (f"{scenario['helper_lowering']}|"
               f"{scenario['false_directive_lowering']}")
        covered = scenario["covered_selected_templates"]
        cap = scenario["homogeneous_recurrent_cap_projection"]
        out[key] = dict(
            covered_slots=covered["mapped_instruction_slots"],
            covered_unmapped=covered["unmapped_operations"],
            cap_slots=cap["mapped_instruction_slots"],
            cap_unmapped=cap["unmapped_operations"],
            cap_opcodes=cap["typed_operation_counts"],
            covered_supplementary=scenario["selected_supplementary_forms"],
            cap_supplementary=scenario["cap_supplementary_forms"],
        )
    return dict(scenarios=out, coverage=forecast["coverage"])


def load_compiles():
    rows = {}
    with open(POST882 / "compiles.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            rows[(row["system"], row["algo"], row["policy"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--skip-systems", default="")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compiles = load_compiles()
    configs = sorted({(s, a) for s, a, _ in compiles})
    if args.configs:
        wanted = {tuple(c.split("/")) for c in args.configs.split(",")}
        configs = [c for c in configs if c in wanted]
    skip = set(filter(None, args.skip_systems.split(",")))
    configs = [c for c in configs if c[0] not in skip]
    compiler = json.loads(COMPILER.read_text())
    policies = args.policies.split(",")
    records = out / "records.jsonl"
    done = set()
    if records.exists():
        for line in records.read_text().splitlines():
            row = json.loads(line)
            done.add((row["system"], row["algo"], row["policy"]))
    for system_name, algo in configs:
        for label in policies:
            key = (system_name, algo, label)
            if key in done:
                continue
            compiled = compiles.get(key)
            if compiled is None:
                continue
            levels = eight_levels(label)
            started = time.perf_counter()
            row = dict(system=system_name, algo=algo, policy=label,
                       levels=list(levels))
            folder = out / "graphs" / f"{system_name}__{algo}__{label}"
            try:
                graph = build_graph(system_name, algo, levels, folder)
                row["graph_seconds"] = time.perf_counter() - started
                wrapper = footprint.construct_typed_body(graph, compiler)
                row["typed_seconds"] = (time.perf_counter() - started
                                        - row["graph_seconds"])
                row.update(summarize(graph, wrapper))
                row["graph_nodes"] = len(graph["nodes"])
                row["status"] = "ok"
            except Exception:
                row["status"] = "error"
                row["error"] = traceback.format_exc()
            row["seconds"] = time.perf_counter() - started
            counts = compiled["sass_counts"]
            row["sass"] = dict(
                instructions=counts["instructions"],
                outside=counts["outside"]["all"],
                back_edges=counts["back_edges"], loops=counts["loops"],
                regs=compiled["regs"], local_bytes=compiled["local_bytes"],
                sass_loops=compiled.get("sass_loops"),
                cubin_sha=compiled["cubin_sha"],
            )
            with open(records, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            summary = row.get("scenarios", {}).get("inline|rolled", {})
            print(f"{system_name:12s} {algo:24s} {label:9s} {row['status']:5s} "
                  f"{row['seconds']:6.1f}s sass {counts['instructions']:6d} "
                  f"covered {summary.get('covered_slots', '-'):>6} "
                  f"cap {summary.get('cap_slots', '-'):>6}", flush=True)


if __name__ == "__main__":
    main()
