#!/usr/bin/env python
"""Paired sweep over direct-LU variants against BiCGSTAB references."""

import argparse
import statistics
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import numpy as np

import cubie as qb
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda

import preconditioner_sweep as ps

precision = np.float32
blocksize = 64

# label -> extra Solver kwargs on top of the model's base settings.
ARM_CONFIGS = (
    ("lu-exact", dict(linear_correction_type="lu")),
    (
        "lu-cached",
        dict(
            linear_correction_type="lu",
            inexact_newton=True,
            prefactored=False,
        ),
    ),
    (
        "lu-prefactored",
        dict(linear_correction_type="lu", inexact_newton=True),
    ),
    (
        "bicgstab-exact",
        dict(
            linear_correction_type="bicgstab",
            preconditioner_type="jacobi",
            preconditioner_order=0,
        ),
    ),
    (
        "bicgstab-cached",
        dict(
            linear_correction_type="bicgstab",
            preconditioner_type="jacobi",
            preconditioner_order=0,
            inexact_newton=True,
        ),
    ),
)

LORENZ_BASE = dict(
    algorithm="kvaerno3",
    atol=1e-06,
    rtol=1e-06,
    save_every=1.0,
    dt_min=1e-12,
    dt_max=1e3,
    output_types=["state"],
    time_logging_level="default",
)

FABBRI_BASE = dict(
    algorithm="kvaerno3",
    atol=1e-06,
    rtol=1e-04,
    save_every=0.01,
    dt_min=1e-12,
    dt_max=1e-2,
    output_types=["state"],
    time_logging_level="default",
)


def occupancy_metrics(solver) -> Dict[str, int]:
    """Return registers, local/shared bytes, and occupancy."""
    (kern,) = solver.kernel.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    first_chunk_runs = int(solver.kernel.run_params[0].runs)
    pad = 4 if solver.kernel.shared_memory_needs_padding else 0
    padded_bytes = solver.kernel.shared_memory_bytes + pad
    dynshared = padded_bytes * min(first_chunk_runs, blocksize)
    actual_blocksize, dynshared = solver.kernel.limit_blocksize(
        blocksize, dynshared, padded_bytes, first_chunk_runs
    )
    dynshared = max(4, dynshared)
    context = cuda.current_context()
    blocks_per_sm = context.get_active_blocks_per_multiprocessor(
        cufunc, actual_blocksize, dynshared
    )
    return {
        "regs": int(kern.regs_per_thread),
        "local_bytes": int(getattr(kern, "local_mem_per_thread", -1)),
        "dynshared": int(dynshared),
        "blocks_per_sm": int(blocks_per_sm),
        "blocksize": int(actual_blocksize),
    }


def factor_sizes(system, tableau) -> Dict[str, str]:
    """Return per-arm cached_aux/lu_factor element counts."""
    exact = system.get_solver_helper("lu_solve")
    cached = system.get_solver_helper("lu_solve", jacobian_at="step")
    prefactored = system.get_solver_helper(
        "lu_solve",
        jacobian_at="step",
        prefactored=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    return {
        "lu-exact": f"factor={exact.lu_nnz}",
        "lu-cached": (
            f"factor={cached.lu_nnz} "
            f"cached_aux={cached.cached_auxiliary_count}"
        ),
        "lu-prefactored": (
            f"cached_aux={prefactored.cached_auxiliary_count}"
        ),
    }


def build_solver(system, base: dict, extra: dict):
    """Return a Solver for one arm's configuration."""
    return qb.Solver(system, **{**base, **extra})


def prepare(system, base, grid_builder, n_runs, duration):
    """Build and warm one solver per arm plus the reference twin."""
    entries = []
    metrics = {}
    arm_list = list(ARM_CONFIGS) + [
        (f"{ARM_CONFIGS[0][0]} (twin)", ARM_CONFIGS[0][1])
    ]
    for name, extra in arm_list:
        start = perf_counter()
        solver = build_solver(system, base, extra)
        inits, params = grid_builder(solver, n_runs)
        ps.solve_once(solver, inits, params, duration)
        metrics[name] = occupancy_metrics(solver)
        print(
            f"built {name}: {perf_counter() - start:.1f} s "
            f"(compile plus one warm-up solve)"
        )
        entries.append((name, solver, inits, params))
    return entries, metrics


def run_sweep(label, entries, metrics, sizes, duration, args):
    """Measure every arm and print paired verdicts with occupancy."""
    kernel_rounds: Dict[str, List[float]] = {
        name: [] for name, *_ in entries
    }
    wall_rounds: Dict[str, List[float]] = {
        name: [] for name, *_ in entries
    }
    for index in range(args.rounds):
        ordered = (
            entries if index % 2 == 0 else list(reversed(entries))
        )
        for name, solver, inits, params in ordered:
            kernel_ms = []
            wall_ms = []
            for _ in range(args.block):
                kernel, wall, _ = ps.solve_once(
                    solver, inits, params, duration
                )
                kernel_ms.append(kernel)
                wall_ms.append(wall)
            kernel_rounds[name].append(
                ps.lowest_mean(kernel_ms, args.min_count)
            )
            wall_rounds[name].append(
                ps.lowest_mean(wall_ms, args.min_count)
            )

    reference = entries[0][0]
    print()
    print(f"=== {label} ===")
    header = (
        f"{'config':<22}{'kernel ms':>11}{'delta %':>9}"
        f"{'wall ms':>11}{'delta %':>9}{'failed':>7}"
        f"{'regs':>6}{'lmem B':>8}{'blk/sm':>7}"
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
        _, _, result = ps.solve_once(solver, inits, params, duration)
        failed, histogram = ps.failure_summary(result)
        meta = metrics[name]
        print(
            f"{name:<22}{kernel_stat:>11.3f}{kernel_delta:>9.2f}"
            f"{wall_stat:>11.3f}{wall_delta:>9.2f}{failed:>7d}"
            f"{meta['regs']:>6d}{meta['local_bytes']:>8d}"
            f"{meta['blocks_per_sm']:>7d}"
        )
        if histogram:
            flags = ", ".join(
                f"{flag}={count}"
                for flag, count in histogram.items()
            )
            print(f"{'':<22}{flags}")
    for name, size_line in sizes.items():
        print(f"per-thread step storage {name}: {size_line}")
    print(
        f"deltas are medians of {args.rounds} paired rounds against "
        f"'{reference}', each the mean of the {args.min_count} lowest "
        f"of {args.block} solves; the (twin) row is the null"
    )


def kvaerno3_tableau():
    """Return the kvaerno3 tableau for factor-size reporting."""
    from cubie.integrators.algorithms.generic_dirk_tableaus import (
        DIRK_TABLEAU_REGISTRY,
    )

    return DIRK_TABLEAU_REGISTRY["kvaerno3"]


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
    parser.add_argument("--n-runs", type=int, default=2**18)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fabbri-runs", type=int, default=1024)
    parser.add_argument("--fabbri-duration", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the requested sweeps."""
    if CUDA_SIMULATION:
        raise SystemExit(
            "lu_variant_sweep.py measures kernel time on a real GPU; "
            "unset NUMBA_ENABLE_CUDASIM"
        )
    args = parse_args(argv)
    tableau = kvaerno3_tableau()
    if args.model in ("lorenz", "both"):
        system = ps.build_lorenz_system()
        sizes = factor_sizes(system, tableau)
        entries, metrics = prepare(
            system,
            LORENZ_BASE,
            ps.lorenz_grid,
            args.n_runs,
            args.duration,
        )
        run_sweep(
            f"lorenz kvaerno3, {args.n_runs} runs, duration "
            f"{args.duration}",
            entries,
            metrics,
            sizes,
            args.duration,
            args,
        )
    if args.model in ("fabbri", "both"):
        system = ps.build_fabbri_system()
        sizes = factor_sizes(system, tableau)
        entries, metrics = prepare(
            system,
            FABBRI_BASE,
            ps.fabbri_grid,
            args.fabbri_runs,
            args.fabbri_duration,
        )
        run_sweep(
            f"fabbri kvaerno3, {args.fabbri_runs} runs, duration "
            f"{args.fabbri_duration}",
            entries,
            metrics,
            sizes,
            args.fabbri_duration,
            args,
        )


if __name__ == "__main__":
    main()
