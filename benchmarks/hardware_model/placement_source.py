"""Construct actual placement-specific implicit graphs without CUDA codegen."""

import argparse
import itertools
import json
from pathlib import Path

from cubie import Solver
from cubie.cache_root import get_cache_root_override, set_cache_root

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model import implicit_source_graph as source
from benchmarks.hardware_model import implicit_workload as workload
from benchmarks.hardware_model.candidate_selection import canonical, file_digest
from benchmarks.hardware_model.workload_identity import workload_identity


def construct(request, output):
    """Capture every local/shared product from fresh actual host factories."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = request["targets"]
    if not targets or len({item["setting"] for item in targets}) != len(targets):
        raise ValueError("Targets need distinct public placement settings")
    names = [item["owner"] + ":" + item["name"] for item in targets]
    if len(set(names)) != len(names):
        raise ValueError("Targets need distinct owner-qualified names")
    result = {}
    previous = get_cache_root_override()
    try:
        for index, spaces in enumerate(itertools.product(
                ("local", "shared"), repeat=len(targets))):
            folder = output / str(index)
            folder.mkdir()
            set_cache_root(folder / "codegen")
            system = placement.SYSTEMS[request["system"]]["build"]()
            kwargs = placement.solver_kwargs(request["system"], request["algo"])
            kwargs.update(request.get("solver_settings", {}))
            kwargs["linear_correction_type"] = workload.PUBLIC_LINEAR_TYPES[
                request["linear_solver"]]
            kwargs.update({item["setting"]: space
                           for item, space in zip(targets, spaces)})
            solver = Solver(system, **kwargs)
            try:
                descriptor = workload.describe_implicit_workload(solver)
                regimes = source.uniform_regime(
                    descriptor, request["newton_bodies"],
                    request["krylov_bodies"])
                graph = source.describe_implicit_source(
                    solver, regimes, request.get("branch_choices", {}))
                identity = dict(zip(names, spaces))
                registry = {item["owner"] + ":" + item["name"]: item
                            for item in graph["registry"]}
                for name, space in identity.items():
                    if registry[name]["declared_location"] != space:
                        raise ValueError("Public setting did not reach target")
                elements = int(solver.kernel.shared_memory_elements)
                padding = int(solver.kernel.shared_memory_needs_padding)
                kernel_file = Path(placement.__file__).resolve().parents[1] / (
                    "src/cubie/batchsolving/BatchSolverKernel.py")
                common = workload_identity(system, solver)
                graph["candidate_construction"] = {
                    "workload_identity": common,
                    "shared_stride_bytes": 4 * (elements + padding),
                    "precision": "float32",
                }
                graph["placement_identity"] = {
                    item["owner"] + ":" + item["name"]:
                    item["declared_location"] for item in graph["registry"]}
                graph["placement_construction"] = {
                    "request": request,
                    "workload_identity": common,
                    "placement_identity": identity,
                    "shared_elements_per_run": elements,
                    "shared_padding_elements": padding,
                    "shared_stride_bytes": 4 * (elements + padding),
                    "precision": "float32",
                    "layout_source": {"path": str(kernel_file),
                                      "sha256": file_digest(kernel_file)},
                    "constructor": {"path": str(Path(__file__).resolve()),
                                    "sha256": file_digest(__file__)},
                }
                graph_path = folder / "graph.json"
                graph_path.write_text(json.dumps(graph, sort_keys=True) + "\n")
                result[canonical(identity)] = {
                    "path": str(graph_path), "sha256": file_digest(graph_path)}
            finally:
                solver.close()
    finally:
        set_cache_root(previous)
    (output / "graphs.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    construct(json.loads(args.request.read_text()), args.output)


if __name__ == "__main__":
    main()
