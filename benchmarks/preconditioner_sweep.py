#!/usr/bin/env python
"""Block-interleaved sweep over preconditioner configurations.

One process builds one solver per configuration on the same system
and ping-pongs solve blocks between them, so every configuration
samples the same GPU clock state. Each block yields the mean of its
lowest ``k`` per-solve kernel times (CUDA events, kernel only) and
the same statistic over wall times; the verdict per configuration is
the median of its per-round percent deltas against the first
configuration listed. A twin of that reference runs last, so its
delta is the machine's null. Each configuration also reports its
failed-run count and the status flags those runs carry.

Two models run:

``lorenz``
    The Lorenz ensemble and radau settings of ``ab_gate.py``'s
    ``adaptive`` config, over the Neumann default and Jacobi orders
    zero, one, and two.
``fabbri``
    The Fabbri-Linder sinoatrial model (35 states) from
    ``tests/fixtures/cellml/Fabbri_Linder.cellml`` on ``radau_iia_5``
    with a BiCGSTAB inner solve, over Jacobi orders zero, one, and
    two. Neumann assumes an identity mass and is not run.

Usage::

    python benchmarks/preconditioner_sweep.py [--model MODEL]
        [--rounds R] [--block N] [--min-count K] [--n-runs N]
        [--duration T] [--fabbri-runs N] [--fabbri-duration T]

All metrics require a real GPU; the script exits under the CUDA
simulator.
"""

import argparse
import contextlib
import io
import statistics
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import cubie as qb
from cubie.cuda_simsafe import CUDA_SIMULATION
from cubie.result_codes import CUBIE_RESULT_CODES

REPO = Path(__file__).resolve().parent.parent
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)

precision = np.float32
blocksize = 64
lorenz_initial_conditions = {"x": 1.0, "y": 0.0, "z": 0.0}

# (label, preconditioner_type, preconditioner_order). The first entry
# is the reference every delta is taken against.
LORENZ_CONFIGS = (
    ("jacobi order 0", "jacobi", 0),
    ("jacobi order 1", "jacobi", 1),
    ("jacobi order 2", "jacobi", 2),
    ("neumann order 2", "neumann", 2),
)
FABBRI_CONFIGS = (
    ("jacobi order 0", "jacobi", 0),
    ("jacobi order 1", "jacobi", 1),
    ("jacobi order 2", "jacobi", 2),
)


def build_lorenz_system():
    """Return the Lorenz system the runtime benchmarks use."""
    return qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name="Lorenz",
        precision=precision,
    )


def build_lorenz_solver(
    system, preconditioner_type: str, preconditioner_order: int
):
    """Return the radau solver of the gate's ``adaptive`` config."""
    return qb.Solver(
        system,
        algorithm="radau",
        preconditioner_type=preconditioner_type,
        preconditioner_order=preconditioner_order,
        atol=1e-06,
        rtol=1e-06,
        save_every=1.0,
        dt_min=1e-12,
        dt_max=1e3,
        step_controller="pid",
        kp=6 / 5,
        kd=0.0,
        ki=0.0,
        max_gain=5.0,
        min_gain=0.1,
        output_types=["state"],
        time_logging_level="default",
    )


FABBRI_PARAMETERS = (
    "Rate_modulation_experiments_ACh",
    "Rate_modulation_experiments_Iso_cas",
)
"""Analog rate-modulation inputs the sweep varies across runs."""


def build_fabbri_system():
    """Return the Fabbri-Linder sinoatrial system."""
    return qb.load_cellml_model(
        str(FABBRI_CELLML),
        precision=precision,
        parameters=list(FABBRI_PARAMETERS),
        voltage_variable="Membrane$V_ode",
    )


def build_fabbri_solver(
    system, preconditioner_type: str, preconditioner_order: int
):
    """Return the radau_iia_5 + BiCGSTAB solver for Fabbri-Linder."""
    return qb.Solver(
        system,
        algorithm="radau_iia_5",
        preconditioner_type=preconditioner_type,
        preconditioner_order=preconditioner_order,
        linear_correction_type="bicgstab",
        atol=1e-06,
        rtol=1e-04,
        save_every=0.01,
        dt_min=1e-12,
        dt_max=1e-2,
        newton_max_iters=5,
        krylov_max_iters=50,
        output_types=["state"],
        time_logging_level="default",
    )


def lorenz_grid(solver, n_runs: int):
    """Return the Lorenz input grid for ``n_runs`` trajectories."""
    return solver.build_grid(
        initial_values=lorenz_initial_conditions,
        parameters={"rho": np.linspace(0.0, 21.0, n_runs)},
    )


def fabbri_grid(solver, n_runs: int):
    """Return an ACh/Iso grid for ``n_runs`` trajectories."""
    side = int(np.ceil(np.sqrt(n_runs)))
    ach, iso = np.meshgrid(
        np.linspace(0.0, 2e-8, side),
        np.linspace(0.0, 1.0, side),
    )
    parameters = {
        FABBRI_PARAMETERS[0]: ach.ravel()[:n_runs],
        FABBRI_PARAMETERS[1]: iso.ravel()[:n_runs],
    }
    return solver.build_grid(parameters=parameters)


def kernel_time_ms(solver) -> float:
    """Return one solve's kernel CUDA-event total in milliseconds."""
    return sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )


def solve_once(solver, inits, params, duration: float):
    """Run one solve; return its ``(kernel_ms, wall_ms, result)``."""
    start = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            initial_values=inits,
            parameters=params,
            blocksize=blocksize,
            duration=duration,
        )
    wall_ms = 1000.0 * (perf_counter() - start)
    return kernel_time_ms(solver), wall_ms, result


def failure_summary(result) -> Tuple[int, Dict[str, int]]:
    """Return the failed-run count and its status-flag histogram."""
    codes = np.asarray(result.status_codes).ravel()
    flags = codes & 0xFFFF
    failed = int(np.count_nonzero(flags))
    histogram: Dict[str, int] = {}
    for member in CUBIE_RESULT_CODES:
        if member.value == 0:
            continue
        count = int(np.count_nonzero(flags & member.value))
        if count:
            histogram[member.name] = count
    return failed, histogram


def lowest_mean(samples: Sequence[float], k: int) -> float:
    """Return the mean of the ``k`` lowest samples."""
    ordered = np.sort(np.asarray(samples))
    return float(ordered[:k].mean())


def run_sweep(
    label: str,
    entries: List[Tuple[str, object, object, object]],
    duration: float,
    rounds: int,
    block: int,
    min_count: int,
) -> None:
    """Measure every configuration and print the paired verdicts.

    Parameters
    ----------
    label
        Model name for the printed heading.
    entries
        ``(config label, solver, inits, params)``, reference first.
    duration
        Solve duration in model time units.
    rounds
        Interleaved rounds; each gives one delta per configuration.
    block
        Solves per configuration per round.
    min_count
        Lowest per-solve times the block statistic averages.
    """
    kernel_rounds: Dict[str, List[float]] = {
        name: [] for name, *_ in entries
    }
    wall_rounds: Dict[str, List[float]] = {
        name: [] for name, *_ in entries
    }
    for index in range(rounds):
        ordered = entries if index % 2 == 0 else list(reversed(entries))
        for name, solver, inits, params in ordered:
            kernel_ms = []
            wall_ms = []
            for _ in range(block):
                kernel, wall, _ = solve_once(
                    solver, inits, params, duration
                )
                kernel_ms.append(kernel)
                wall_ms.append(wall)
            kernel_rounds[name].append(
                lowest_mean(kernel_ms, min_count)
            )
            wall_rounds[name].append(lowest_mean(wall_ms, min_count))

    reference = entries[0][0]
    print()
    print(f"=== {label} ===")
    header = (
        f"{'config':<24}{'kernel ms':>12}{'delta %':>10}"
        f"{'wall ms':>12}{'delta %':>10}{'failed':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, solver, inits, params in entries:
        kernel_stat = statistics.median(kernel_rounds[name])
        wall_stat = statistics.median(wall_rounds[name])
        kernel_delta = statistics.median(
            [
                100.0 * (value / base - 1.0)
                for value, base in zip(
                    kernel_rounds[name], kernel_rounds[reference]
                )
            ]
        )
        wall_delta = statistics.median(
            [
                100.0 * (value / base - 1.0)
                for value, base in zip(
                    wall_rounds[name], wall_rounds[reference]
                )
            ]
        )
        _, _, result = solve_once(solver, inits, params, duration)
        failed, histogram = failure_summary(result)
        print(
            f"{name:<24}{kernel_stat:>12.3f}{kernel_delta:>10.2f}"
            f"{wall_stat:>12.3f}{wall_delta:>10.2f}{failed:>9d}"
        )
        if histogram:
            flags = ", ".join(
                f"{flag}={count}" for flag, count in histogram.items()
            )
            print(f"{'':<24}{flags}")
    print(
        f"deltas are medians of {rounds} paired rounds against "
        f"'{reference}', each the mean of the {min_count} lowest of "
        f"{block} solves; the (twin) row is the same configuration "
        f"built twice, so its delta is the null"
    )


def prepare(
    builder,
    system,
    configs,
    grid_builder,
    n_runs: int,
    duration: float,
) -> List[Tuple[str, object, object, object]]:
    """Build, load, and warm one solver per configuration.

    A twin of the reference configuration is built last, so the
    table carries the null for this machine and block shape.
    """
    entries = []
    twin = (f"{configs[0][0]} (twin)",) + tuple(configs[0][1:])
    for name, precond_type, order in list(configs) + [twin]:
        start = perf_counter()
        solver = builder(system, precond_type, order)
        inits, params = grid_builder(solver, n_runs)
        solve_once(solver, inits, params, duration)
        print(
            f"built {name}: {perf_counter() - start:.1f} s "
            f"(compile plus one warm-up solve)"
        )
        entries.append((name, solver, inits, params))
    return entries


def parse_args(argv: Optional[Sequence[str]] = None):
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("lorenz", "fabbri", "both"),
        default="both",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block", type=int, default=9)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--n-runs", type=int, default=2**20)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fabbri-runs", type=int, default=1024)
    parser.add_argument("--fabbri-duration", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the requested sweeps."""
    if CUDA_SIMULATION:
        raise SystemExit(
            "preconditioner_sweep.py measures kernel time on a real "
            "GPU; unset NUMBA_ENABLE_CUDASIM"
        )
    args = parse_args(argv)
    if args.model in ("lorenz", "both"):
        system = build_lorenz_system()
        entries = prepare(
            build_lorenz_solver,
            system,
            LORENZ_CONFIGS,
            lorenz_grid,
            args.n_runs,
            args.duration,
        )
        run_sweep(
            f"lorenz radau, {args.n_runs} runs, duration "
            f"{args.duration}",
            entries,
            args.duration,
            args.rounds,
            args.block,
            args.min_count,
        )
    if args.model in ("fabbri", "both"):
        system = build_fabbri_system()
        entries = prepare(
            build_fabbri_solver,
            system,
            FABBRI_CONFIGS,
            fabbri_grid,
            args.fabbri_runs,
            args.fabbri_duration,
        )
        run_sweep(
            f"fabbri radau_iia_5 + bicgstab, {args.fabbri_runs} runs, "
            f"duration {args.fabbri_duration}",
            entries,
            args.fabbri_duration,
            args.rounds,
            args.block,
            args.min_count,
        )


if __name__ == "__main__":
    main()
