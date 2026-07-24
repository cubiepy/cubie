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


def kernel_time_ms(solver) -> tuple[float, int]:
    """Return total kernel time and launch count for the last solve."""

    events = [
        event for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    ]
    return sum(event.elapsed_time_ms() for event in events), len(events)


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


def tune_algorithm(
    system,
    algorithm: str,
    problem: str,
    target_ms: float,
    floor_improvement: float,
    max_n: int,
):
    """Tune one algorithm and return its final hot-launch state."""

    solver = qb.Solver(system, **solver_kwargs(algorithm))
    n = 32
    previous_per_run = None
    floor_streak = 0
    history = []
    final = None
    while True:
        if n > max_n:
            raise RuntimeError(
                f"{algorithm} reached the --max-n safety guard "
                f"({max_n}) before either the {target_ms:g} ms target "
                "or the throughput floor; increase --max-n"
            )
        inits, params, drivers = inputs_for(solver, problem, n)
        solve_once(solver, inits, params, drivers)
        samples = []
        result = None
        for _ in range(3):
            result = solve_once(solver, inits, params, drivers)
            kernel_ms, launch_count = kernel_time_ms(solver)
            if launch_count != 1:
                raise RuntimeError(
                    f"{algorithm} chunked into {launch_count} launches "
                    f"at n={n}; the NCU capture requires one launch"
                )
            samples.append(kernel_ms)
        measured_ms = min(samples)
        per_run = measured_ms / n
        improvement = None
        if previous_per_run is not None:
            improvement = 1.0 - per_run / previous_per_run
            if improvement < floor_improvement:
                floor_streak += 1
            else:
                floor_streak = 0
        history.append(
            {
                "n": n,
                "kernel_ms": measured_ms,
                "ms_per_trajectory": per_run,
                "improvement": improvement,
            }
        )
        print(
            f"@TUNE {algorithm} n={n} kernel_ms={measured_ms:.6f} "
            f"ns_per_trajectory={1e6 * per_run:.3f}",
            flush=True,
        )
        final = (solver, inits, params, drivers, result)
        if measured_ms >= target_ms or floor_streak >= 2:
            break
        previous_per_run = per_run
        n *= 2
    solver, inits, params, drivers, result = final
    selected_n = history[-1]["n"]
    if history[-1]["kernel_ms"] >= target_ms:
        stop_reason = "target"
    else:
        stop_reason = "throughput-floor"
    record = {
        "n": selected_n,
        "kernel_ms": history[-1]["kernel_ms"],
        "ns_per_trajectory": (
            1e6 * history[-1]["ms_per_trajectory"]
        ),
        "stop_reason": stop_reason,
        "tuning": history,
    }
    record.update(counter_summary(result))
    return solver, inits, params, drivers, record


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
    parser.add_argument("--target-ms", type=float, required=True)
    parser.add_argument("--floor-improvement", type=float, required=True)
    parser.add_argument("--max-n", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Calibrate four solvers and expose exactly four hot launches."""

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
    system = build_problem(args.problem)
    launches = {}
    records = {}
    for algorithm in ALGORITHMS:
        solver, inits, params, drivers, record = tune_algorithm(
            system,
            algorithm,
            args.problem,
            args.target_ms,
            args.floor_improvement,
            args.max_n,
        )
        launches[algorithm] = (solver, inits, params, drivers)
        records[algorithm] = record
    for algorithm in ALGORITHMS:
        solve_once(*launches[algorithm])
    manifest = {
        "manifest_version": 1,
        "problem": args.problem,
        "backend": args.backend,
        "algorithms": records,
        "launch_order": list(ALGORITHMS),
    }
    prefix = f"{args.problem}_{args.backend}"
    manifest_path = args.output_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("@PROFILE_START", flush=True)
    cuda.profile_start()
    for algorithm in ALGORITHMS:
        print(f"@PROFILE {algorithm}", flush=True)
        solve_once(*launches[algorithm])
    cuda.profile_stop()
    print("@PROFILE_STOP", flush=True)
    for solver, _, _, _ in launches.values():
        solver.close()


if __name__ == "__main__":
    main()
