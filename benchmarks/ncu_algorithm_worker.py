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


def wave_launch_geometry(solver, waves: int) -> dict[str, int]:
    """Return occupancy-aware launch geometry for ``waves`` waves."""

    (compiled_kernel,) = solver.kernel.kernel.overloads.values()
    if hasattr(compiled_kernel, "_ensure_kernel_attrs"):
        compiled_kernel._ensure_kernel_attrs()
    cuda_function = compiled_kernel._codelibrary.get_cufunc()

    seed_runs = int(solver.kernel.run_params[0].runs)
    pad = 4 if solver.kernel.shared_memory_needs_padding else 0
    padded_bytes = solver.kernel.shared_memory_bytes + pad
    dynamic_shared_memory = padded_bytes * min(seed_runs, BLOCKSIZE)
    actual_blocksize, dynamic_shared_memory = (
        solver.kernel.limit_blocksize(
            BLOCKSIZE,
            dynamic_shared_memory,
            padded_bytes,
            seed_runs,
        )
    )
    dynamic_shared_memory = max(4, dynamic_shared_memory)
    threads_per_trajectory = (
        solver.kernel.single_integrator.threads_per_step
    )
    trajectories_per_block = (
        actual_blocksize // threads_per_trajectory
    )
    if threads_per_trajectory != 1:
        raise RuntimeError(
            "the NCU wave estimate requires one thread per trajectory; "
            f"received {threads_per_trajectory}"
        )

    context = cuda.current_context()
    blocks_per_multiprocessor = (
        context.get_active_blocks_per_multiprocessor(
            cuda_function,
            actual_blocksize,
            dynamic_shared_memory,
        )
    )
    if blocks_per_multiprocessor < 1:
        raise RuntimeError(
            "the kernel cannot launch: occupancy query reported "
            f"{blocks_per_multiprocessor} resident blocks per "
            "multiprocessor"
        )
    multiprocessors = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid_blocks = (
        waves * multiprocessors * blocks_per_multiprocessor
    )
    n = grid_blocks * trajectories_per_block
    return {
        "n": int(n),
        "waves": waves,
        "multiprocessors": int(multiprocessors),
        "blocks_per_multiprocessor": int(
            blocks_per_multiprocessor
        ),
        "trajectories_per_block": int(trajectories_per_block),
        "blocksize": int(actual_blocksize),
        "dynamic_shared_memory_bytes": int(dynamic_shared_memory),
        "grid_blocks": int(grid_blocks),
    }


def prepare_launch(
    label: str,
    algorithm: str,
    problem: str,
    waves: int,
    lto: bool,
    native_timing: bool,
):
    """Prepare one occupancy-sized launch for one algorithm/LTO arm.

    Each launch builds its own system instance: two Solvers sharing
    one SymbolicODE mutate shared factory state and produce
    build-order-dependent kernels.

    The seed solve at ``BLOCKSIZE`` trajectories forces compilation
    and populates the run parameters the occupancy query reads; the
    full-size solve validates the single-launch constraint and
    collects the iteration counters. ``native_timing`` adds two more
    solves for a min-of-3 CUDA-event kernel time — meaningful only
    outside NCU, whose replay overhead contaminates native timing.
    """

    system = build_problem(problem)
    solver = qb.Solver(system, **solver_kwargs(algorithm))
    # The MLIR backend's LTO link strips line tables, so per-line
    # source attribution requires the lto=False arm; the lto=True arm
    # profiles the production build.
    solver.update(lto=lto, silent=True)
    seed_inits, seed_params, seed_drivers = inputs_for(
        solver, problem, BLOCKSIZE
    )
    solve_once(solver, seed_inits, seed_params, seed_drivers)
    geometry = wave_launch_geometry(solver, waves)
    n = geometry["n"]
    inits, params, drivers = inputs_for(solver, problem, n)

    samples = []
    result = None
    for _ in range(3 if native_timing else 1):
        result = solve_once(solver, inits, params, drivers)
        kernel_ms, launch_count = kernel_time_ms(solver)
        if launch_count != 1:
            raise RuntimeError(
                f"{label} chunked into {launch_count} launches "
                f"at n={n}; the NCU capture requires one launch"
            )
        samples.append(kernel_ms)
    record = {
        **geometry,
        "algorithm": algorithm,
        "lto": lto,
    }
    timing_note = ""
    if native_timing:
        measured_ms = min(samples)
        per_trajectory = 1e6 * measured_ms / n
        record["kernel_ms"] = measured_ms
        record["ns_per_trajectory"] = per_trajectory
        timing_note = (
            f" kernel_ms={measured_ms:.6f} "
            f"ns_per_trajectory={per_trajectory:.3f}"
        )
    print(
        f"@SIZE {label} waves={waves} n={n} "
        f"grid_blocks={geometry['grid_blocks']} "
        f"blocks_per_sm={geometry['blocks_per_multiprocessor']}"
        f"{timing_note}",
        flush=True,
    )
    record.update(counter_summary(result))
    return solver, inits, params, drivers, record


def launch_plan(
    algorithms: Sequence[str],
    lto_mode: str,
) -> tuple[tuple[str, str, bool], ...]:
    """Return (label, algorithm, lto) launches in profile order.

    ``both`` profiles the two arms of each algorithm adjacently so
    their launches share clock state in the capture.
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
    parser.add_argument("--waves", type=int, required=True)
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
    parser.add_argument(
        "--native-timing",
        action="store_true",
        help=(
            "record min-of-3 CUDA-event kernel times in the manifest; "
            "direct mode only — NCU replay contaminates native timing"
        ),
    )
    args = parser.parse_args(argv)
    if args.waves < 1:
        parser.error("--waves must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Size the selected solvers and expose one hot launch each."""

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
            args.waves,
            lto,
            args.native_timing,
        )
        launches[label] = (solver, inits, params, drivers)
        records[label] = record
    manifest = {
        "problem": args.problem,
        "backend": args.backend,
        "waves": args.waves,
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
