#!/usr/bin/env python
"""Internal hot-kernel worker for ``ncu_algorithm_comparison.py``."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    raise SystemExit("the NCU worker requires a real GPU")

import cubie as qb  # noqa: E402
from cubie.cuda_simsafe import cuda  # noqa: E402
from cubie.time_logger import default_timelogger  # noqa: E402
from tests.system_fixtures import (  # noqa: E402
    build_lorenz_julia_system,
    build_three_state_very_stiff_system,
)


ALGORITHMS = ("tsit5", "kvaerno3", "radau", "ode23s")
PRECISION = np.float32
BLOCKSIZE = 64
# Fixed batch size: fills at least ten occupancy waves at up to 24
# resident blocks per SM on a 56-SM GPU (the highest occupancy any of
# these kernels reaches). Edit when hardware or occupancy changes
# significantly.
N_TRAJECTORIES = 2**20
COUNTER_NAMES = (
    "newton",
    "krylov",
    "attempted",
    "rejected",
)


def solver_kwargs(algorithm: str) -> dict[str, object]:
    """Return the Lorenz benchmark's adaptive settings."""

    settings = {
        "algorithm": algorithm,
        "atol": 1e-6,
        "rtol": 1e-6,
        "save_every": 1.0,
        "dt_min": 1e-12,
        "dt_max": 1e3,
        "step_controller": "pid",
        "kp": 6 / 5,
        "kd": 0.0,
        "ki": 0.0,
        "max_gain": 5.0,
        "min_gain": 0.1,
        "output_types": ["state", "iteration_counters"],
        "time_logging_level": "default",
        "lineinfo": True,
    }
    if algorithm != "tsit5":
        settings["preconditioner_type"] = "jacobi"
    return settings


def build_problem(problem: str):
    """Return the selected benchmark system."""

    if problem == "lorenz":
        return build_lorenz_julia_system(PRECISION)
    if problem == "very-stiff":
        return build_three_state_very_stiff_system(PRECISION)
    raise ValueError(f"unknown problem {problem!r}")


def inputs_for(solver, problem: str, n: int):
    """Return the selected problem's verbatim batch and zero driver."""

    if problem == "lorenz":
        inits, params = solver.build_grid(
            initial_values={"x": 1.0, "y": 0.0, "z": 0.0},
            parameters={
                "rho": np.linspace(
                    0.0, 21.0, n, dtype=PRECISION
                )
            },
        )
        return inits, params, None
    inits, params = solver.build_grid(
        initial_values={
            "x0": np.full(n, 0.5, dtype=PRECISION),
            "x1": np.full(n, 0.25, dtype=PRECISION),
            "x2": np.full(n, 0.1, dtype=PRECISION),
        },
        parameters={
            "k1": np.full(n, 150.0, dtype=PRECISION),
            "k2": np.full(n, 900.0, dtype=PRECISION),
            "k3": np.full(n, 1200.0, dtype=PRECISION),
            "n0": np.full(n, 40.0, dtype=PRECISION),
            "n1": np.full(n, 30.0, dtype=PRECISION),
            "n2": np.full(n, 20.0, dtype=PRECISION),
        },
    )
    drivers = {
        "d0": np.zeros(4, dtype=PRECISION),
        "driver_sample_period": PRECISION(1.0 / 3.0),
    }
    return inits, params, drivers


def solve_once(solver, inits, params, drivers):
    """Run and return one benchmark solve."""

    return solver.solve(
        initial_values=inits,
        parameters=params,
        drivers=drivers,
        blocksize=BLOCKSIZE,
        duration=1.0,
    )


def launch_count(solver) -> int:
    """Return the number of kernel launches in the last solve."""

    return sum(
        1 for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def counter_summary(result) -> dict[str, float]:
    """Return final cumulative counters per trajectory."""

    counters = np.asarray(result.iteration_counters)
    final = counters[-1]
    summary = {}
    for index, name in enumerate(COUNTER_NAMES):
        summary[f"{name}_per_trajectory"] = float(
            np.mean(final[index])
        )
    summary["accepted_per_trajectory"] = (
        summary["attempted_per_trajectory"]
        - summary["rejected_per_trajectory"]
    )
    return summary


def prepare_launch(label: str, algorithm: str, problem: str, lto: bool):
    """Prepare one launch for one algorithm/LTO arm.

    Each arm builds its own system instance: two Solvers sharing one
    SymbolicODE mutate shared factory state and produce
    build-order-dependent kernels. The single preparation solve
    compiles the kernel outside the profile range, checks that the
    batch ran as one launch (the memory manager silently splits
    batches it cannot fit, which would break the launch-to-label
    mapping in the capture), and collects the iteration counters.
    """

    system = build_problem(problem)
    solver = qb.Solver(system, **solver_kwargs(algorithm))
    # The MLIR backend's LTO link strips line tables, so per-line
    # source attribution requires the lto=False arm; the lto=True arm
    # profiles the production build.
    solver.update(lto=lto, silent=True)
    inits, params, drivers = inputs_for(
        solver, problem, N_TRAJECTORIES
    )
    result = solve_once(solver, inits, params, drivers)
    launches = launch_count(solver)
    if launches != 1:
        raise RuntimeError(
            f"{label} chunked into {launches} launches at "
            f"n={N_TRAJECTORIES}; the NCU capture requires one launch"
        )
    record = {
        "algorithm": algorithm,
        "lto": lto,
        "n": N_TRAJECTORIES,
    }
    record.update(counter_summary(result))
    print(f"@PREP {label} n={N_TRAJECTORIES}", flush=True)
    return solver, inits, params, drivers, record


def launch_plan(
    algorithms: Sequence[str],
    lto_mode: str,
) -> tuple[tuple[str, str, bool], ...]:
    """Return (label, algorithm, lto) launches in profile order.

    ``both`` profiles the two arms of each algorithm adjacently so
    their launches sit side by side in the capture.
    """

    arms = {"on": (True,), "off": (False,), "both": (True, False)}[
        lto_mode
    ]
    plan = []
    for algorithm in algorithms:
        for lto in arms:
            if len(arms) == 1:
                label = algorithm
            else:
                label = f"{algorithm}-{'lto' if lto else 'nolto'}"
            plan.append((label, algorithm, lto))
    return tuple(plan)


def selected_algorithms(value: str) -> tuple[str, ...]:
    """Return the requested algorithms in canonical launch order."""

    names = {name.strip() for name in value.split(",") if name.strip()}
    unknown = names - set(ALGORITHMS)
    if unknown:
        raise ValueError(
            f"unknown algorithms {sorted(unknown)}; "
            f"expected a comma-separated subset of {ALGORITHMS}"
        )
    if not names:
        raise ValueError("no algorithms selected")
    return tuple(name for name in ALGORITHMS if name in names)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse worker arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        required=True,
        choices=("lorenz", "very-stiff"),
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("numba-cuda", "mlir"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithms",
        required=True,
        type=selected_algorithms,
        help="comma-separated subset of " + ",".join(ALGORITHMS),
    )
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--lto",
        required=True,
        choices=("on", "off", "both"),
        help=(
            "LTO arms to profile: on (production build), off "
            "(per-line source attribution), or both in succession"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Prepare the selected solvers and profile one launch each."""

    args = parse_args(argv)
    active_backend = os.environ.get("CUBIE_CUDA_BACKEND")
    if active_backend != args.backend:
        raise SystemExit(
            f"worker requested {args.backend}, environment has "
            f"{active_backend!r}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_timelogger.set_verbosity("default")
    qb.default_memmgr.set_limit_mode("active")
    plan = launch_plan(args.algorithms, args.lto)
    launches = {}
    records = {}
    for label, algorithm, lto in plan:
        solver, inits, params, drivers, record = prepare_launch(
            label,
            algorithm,
            args.problem,
            lto,
        )
        launches[label] = (solver, inits, params, drivers)
        records[label] = record
    manifest = {
        "problem": args.problem,
        "backend": args.backend,
        "lto": args.lto,
        "algorithms": records,
        "launch_order": [label for label, _, _ in plan],
    }
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("@PROFILE_START", flush=True)
    cuda.profile_start()
    for label, _, _ in plan:
        print(f"@PROFILE {label}", flush=True)
        solve_once(*launches[label])
    cuda.profile_stop()
    print("@PROFILE_STOP", flush=True)
    for solver, _, _, _ in launches.values():
        solver.close()


if __name__ == "__main__":
    main()
