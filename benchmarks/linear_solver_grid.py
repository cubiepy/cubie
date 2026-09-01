#!/usr/bin/env python
"""Timing grid over linear solvers, preconditioners, and variants.

Four systems (lorenz, lorenz96 n=10/20, fabbri) x three implicit
algorithms (rosenbrock23, kvaerno3, radau_iia_5). Per group, part 1
sweeps BiCGSTAB preconditioners (jacobi 0-2, none);
part 2 crosses minimal_residual/bicgstab (winning preconditioner)
and lu with each supported Newton variant (exact, inexact,
prefactored; rosenbrock23 is always cached). Configurations in one
group block-interleave over a shared input grid, as in
``preconditioner_sweep.py``: block statistic = mean of the
``min_count`` lowest per-solve kernel times, verdict = median
per-round delta against the group reference, twin row = the null.
The trajectory count is two waves at the device occupancy ceiling
(rounded up to a power of two) so every kernel time is measured on
a saturated device; achieved waves print per config after warm-up.
Rows append to ``--csv`` as each sweep completes. Requires a real
GPU; exits under the CUDA simulator.

Usage::

    python benchmarks/linear_solver_grid.py [--systems ...]
        [--algorithms ...] [--rounds R] [--block N] [--min-count K]
        [--n-runs N] [--fabbri-runs N] [--csv PATH] [--smoke]
        [--timeout-factor F]
"""

import argparse
import contextlib
import csv
import io
import math
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import cubie as qb
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda
from cubie.result_codes import CUBIE_RESULT_CODES

REPO = Path(__file__).resolve().parent.parent
FABBRI_CELLML = (
    REPO / "tests" / "fixtures" / "cellml" / "Fabbri_Linder.cellml"
)

precision = np.float32
blocksize = 64

ALGORITHMS = ("rosenbrock23", "kvaerno3", "radau_iia_5")

# (label, type, order); the first entry is the part-1 reference.
PART1_PRECONDITIONERS = (
    ("jacobi-0", "jacobi", 0),
    ("jacobi-1", "jacobi", 1),
    ("jacobi-2", "jacobi", 2),
    ("none", "none", 0),
)


def part1_preconditioners(algorithm: str) -> list:
    """Return part 1 rows legal for ``algorithm``."""
    rows = list(PART1_PRECONDITIONERS)
    if algorithm == "radau_iia_5":
        # Jacobi series orders diverge on stacked FIRK operators.
        rows = [
            row for row in rows
            if not (row[1] == "jacobi" and row[2] > 0)
        ]
    return rows

FABBRI_PARAMETERS = (
    "Rate_modulation_experiments_ACh",
    "Rate_modulation_experiments_Iso_cas",
)


def build_lorenz_system():
    """Return the ab-gate Lorenz system."""
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


def build_lorenz96_system(n: int):
    """Return cyclic Lorenz 96 (GPUODEBenchmarks) with ``n`` states."""
    lines = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        lines.append(
            "dx{0} = (x{1} - x{2}) * x{3} - x{0} + F".format(
                i, ip1, im2, im1
            )
        )
    states = {
        "x{0}".format(i): 9.0 if i == 1 else 8.0
        for i in range(1, n + 1)
    }
    return qb.create_ODE_system(
        "\n".join(lines),
        states=states,
        parameters={"F": 8.0},
        name=f"Lorenz96_{n}",
        precision=precision,
    )


def build_fabbri_system():
    """Return the Fabbri-Linder sinoatrial system."""
    return qb.load_cellml_model(
        str(FABBRI_CELLML),
        precision=precision,
        parameters=list(FABBRI_PARAMETERS),
        voltage_variable="Membrane$V_ode",
    )


def lorenz_grid(solver, n_runs: int):
    """Return the ab-gate rho sweep for ``n_runs`` trajectories."""
    return solver.build_grid(
        initial_values={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": np.linspace(0.0, 21.0, n_runs)},
    )


def lorenz96_grid(solver, n_runs: int):
    """Return the GPUODEBenchmarks F sweep for ``n_runs``."""
    return solver.build_grid(
        parameters={"F": np.linspace(0.0, 16.0, n_runs)},
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


# Tolerances/durations follow ab_gate (lorenz), GPUODEBenchmarks
# (lorenz96), and preconditioner_sweep with end-point saves
# (fabbri); step controllers stay on each algorithm's defaults.
SYSTEMS = {
    "lorenz": {
        "build": build_lorenz_system,
        "grid": lorenz_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "lorenz96_10": {
        "build": lambda: build_lorenz96_system(10),
        "grid": lorenz96_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "lorenz96_20": {
        "build": lambda: build_lorenz96_system(20),
        "grid": lorenz96_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-06,
            "dt_min": 1e-12,
            "dt_max": 1e3,
        },
    },
    "fabbri": {
        "build": build_fabbri_system,
        "grid": fabbri_grid,
        "duration": 1.0,
        "solver_kwargs": {
            "atol": 1e-06,
            "rtol": 1e-04,
            "dt_min": 1e-12,
            "dt_max": 1e-2,
        },
        "constants": {"Rate_modulation_experiments_ANS": 1.0},
    },
}


class Config:
    """One measured configuration: a solver plus its labelling."""

    def __init__(self, label, solver, variant, correction, precond,
                 order):
        self.label = label
        self.solver = solver
        self.variant = variant
        self.correction = correction
        self.precond = precond
        self.order = order
        self.waves = float("nan")


def build_solver(system, algorithm, spec, correction, precond,
                 order, variant):
    """Return one configured ``qb.Solver`` for a grid row."""
    kwargs = dict(spec["solver_kwargs"])
    kwargs.update(
        algorithm=algorithm,
        linear_correction_type=correction,
        output_types=["state"],
        time_logging_level="default",
    )
    if correction != "lu":
        kwargs["preconditioner_type"] = precond
        kwargs["preconditioner_order"] = order
    if algorithm != "rosenbrock23":
        if variant == "inexact":
            kwargs["inexact_newton"] = True
            kwargs["prefactored"] = False
        elif variant == "prefactored":
            kwargs["inexact_newton"] = True
            kwargs["prefactored"] = True
    solver = qb.Solver(system, **kwargs)
    constants = spec.get("constants")
    if constants:
        solver.update(constants)
    return solver


def solve_once(solver, inits, params, duration: float):
    """Run one solve; return ``(kernel_ms, wall_ms, result)``."""
    start = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            initial_values=inits,
            parameters=params,
            blocksize=blocksize,
            duration=duration,
        )
    wall_ms = 1000.0 * (perf_counter() - start)
    kernel_ms = sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )
    return kernel_ms, wall_ms, result


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


def achieved_waves(solver, n_runs: int) -> float:
    """Return occupancy waves filled at the actual launch geometry."""
    (kern,) = solver.kernel.kernel.overloads.values()
    if hasattr(kern, "_ensure_kernel_attrs"):
        kern._ensure_kernel_attrs()
    cufunc = kern._codelibrary.get_cufunc()
    kernel = solver.kernel
    if hasattr(kernel, "launch_blocksize"):
        actual_blocksize = kernel.launch_blocksize
        dynshared = 0
        runs_per_block = kernel.runs_per_block
    else:
        first_chunk_runs = int(kernel.run_params[0].runs)
        pad = 4 if kernel.shared_memory_needs_padding else 0
        padded_bytes = kernel.shared_memory_bytes + pad
        dynshared = padded_bytes * min(first_chunk_runs, blocksize)
        actual_blocksize, dynshared = kernel.limit_blocksize(
            blocksize, dynshared, padded_bytes, first_chunk_runs
        )
        dynshared = max(4, dynshared)
        threads_per_loop = kernel.single_integrator.threads_per_step
        runs_per_block = actual_blocksize // threads_per_loop
    context = cuda.current_context()
    blocks_per_sm = context.get_active_blocks_per_multiprocessor(
        cufunc, actual_blocksize, dynshared
    )
    device = cuda.get_current_device()
    total_blocks = math.ceil(n_runs / runs_per_block)
    resident = blocks_per_sm * device.MULTIPROCESSOR_COUNT
    return total_blocks / resident


def default_n_runs() -> int:
    """Two waves at the occupancy ceiling, next power of two up."""
    device = cuda.get_current_device()
    max_threads = getattr(
        device,
        "MAX_THREADS_PER_MULTI_PROCESSOR",
        getattr(device, "MAX_THREADS_PER_MULTIPROCESSOR", 2048),
    )
    floor = 2 * int(device.MULTIPROCESSOR_COUNT) * int(max_threads)
    return 2 ** math.ceil(math.log2(floor))


def lowest_mean(samples: Sequence[float], k: int) -> float:
    """Return the mean of the ``k`` lowest samples."""
    ordered = np.sort(np.asarray(samples))
    return float(ordered[:k].mean())


def probe_config(entry: Config, system_name: str, algorithm: str,
                 n_runs: int, timeout_s: float,
                 grace_s: float) -> bool:
    """Trial-solve ``entry`` in a killable child; False on timeout."""
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--probe", system_name, algorithm, entry.correction,
        entry.precond, str(entry.order), entry.variant,
        "--n-runs", str(n_runs),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    marks: Dict[str, float] = {}

    def read_marks():
        for line in process.stdout:
            line = line.strip()
            if line in ("SETUP_DONE", "PROBE_OK"):
                marks[line] = perf_counter()

    reader = threading.Thread(target=read_marks, daemon=True)
    reader.start()
    start = perf_counter()
    while process.poll() is None:
        now = perf_counter()
        setup_done = marks.get("SETUP_DONE")
        expired = (
            now - setup_done > timeout_s
            if setup_done is not None and "PROBE_OK" not in marks
            else setup_done is None and now - start > grace_s
        )
        if expired:
            process.kill()
            process.wait()
            return False
        sleep(0.5)
    return True


def prepare(entries: List[Config], inits, params, duration: float,
            n_runs: int, system_name: str, algorithm: str,
            timeout_factor: float, setup_s: float,
            ) -> Tuple[List[Config], List[Config]]:
    """Warm configurations; return the (kept, timed-out) split."""
    kept: List[Config] = []
    dropped: List[Config] = []
    reference_warm = 0.0
    for entry in entries:
        probed = kept and not entry.label.endswith("(twin)")
        if probed:
            timeout_s = timeout_factor * reference_warm
            grace_s = setup_s + 120.0 + timeout_s
            if not probe_config(
                entry, system_name, algorithm, n_runs, timeout_s,
                grace_s,
            ):
                print(
                    f"  {entry.label}: probe exceeded "
                    f"{timeout_s:.0f} s, dropped"
                )
                dropped.append(entry)
                continue
        start = perf_counter()
        solve_once(entry.solver, inits, params, duration)
        entry.waves = achieved_waves(entry.solver, n_runs)
        warm = perf_counter() - start
        if not kept:
            reference_warm = warm
        note = "" if entry.waves >= 2.0 else "  ** UNDER 2 WAVES **"
        print(
            f"  built {entry.label}: {warm:.1f} s, "
            f"{entry.waves:.1f} waves{note}"
        )
        kept.append(entry)
    return kept, dropped


def dropped_row(entry: Config) -> dict:
    """Return the CSV row recording a probe-timeout drop."""
    return {
        "config": entry.label,
        "variant": entry.variant,
        "correction": entry.correction,
        "preconditioner": entry.precond,
        "preconditioner_order": entry.order,
        "kernel_ms": float("nan"),
        "kernel_delta_pct": float("nan"),
        "wall_ms": float("nan"),
        "waves": float("nan"),
        "failed": -1,
        "flags": "PROBE_TIMEOUT",
    }


def run_sweep(heading: str, entries: List[Config], inits, params,
              duration: float, rounds: int, block: int,
              min_count: int) -> List[dict]:
    """Measure one interleaved group; print and return its rows."""
    kernel_rounds: Dict[str, List[float]] = {
        entry.label: [] for entry in entries
    }
    wall_rounds: Dict[str, List[float]] = {
        entry.label: [] for entry in entries
    }
    failures: Dict[str, Tuple[int, Dict[str, int]]] = {}
    for index in range(rounds):
        ordered = entries if index % 2 == 0 else list(reversed(entries))
        for entry in ordered:
            kernel_ms = []
            wall_ms = []
            for _ in range(block):
                kernel, wall, result = solve_once(
                    entry.solver, inits, params, duration
                )
                kernel_ms.append(kernel)
                wall_ms.append(wall)
                if entry.label not in failures:
                    failures[entry.label] = failure_summary(result)
            kernel_rounds[entry.label].append(
                lowest_mean(kernel_ms, min_count)
            )
            wall_rounds[entry.label].append(
                lowest_mean(wall_ms, min_count)
            )
            print(
                f"  round {index + 1}/{rounds} {entry.label}: "
                f"{kernel_rounds[entry.label][-1]:.3f} ms",
                flush=True,
            )

    reference = entries[0].label
    rows = []
    print()
    print(f"=== {heading} ===")
    header = (
        f"{'config':<34}{'kernel ms':>12}{'delta %':>10}"
        f"{'wall ms':>12}{'waves':>7}{'failed':>9}"
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        kernel_stat = statistics.median(kernel_rounds[entry.label])
        wall_stat = statistics.median(wall_rounds[entry.label])
        kernel_delta = statistics.median(
            [
                100.0 * (value / base - 1.0)
                for value, base in zip(
                    kernel_rounds[entry.label],
                    kernel_rounds[reference],
                )
            ]
        )
        failed, histogram = failures[entry.label]
        print(
            f"{entry.label:<34}{kernel_stat:>12.3f}"
            f"{kernel_delta:>10.2f}{wall_stat:>12.3f}"
            f"{entry.waves:>7.1f}{failed:>9d}"
        )
        if histogram:
            flags = ", ".join(
                f"{flag}={count}"
                for flag, count in histogram.items()
            )
            print(f"{'':<34}{flags}")
        rows.append(
            {
                "config": entry.label,
                "variant": entry.variant,
                "correction": entry.correction,
                "preconditioner": entry.precond,
                "preconditioner_order": entry.order,
                "kernel_ms": kernel_stat,
                "kernel_delta_pct": kernel_delta,
                "wall_ms": wall_stat,
                "waves": entry.waves,
                "failed": failed,
                "flags": ";".join(
                    f"{flag}={count}"
                    for flag, count in histogram.items()
                ),
            }
        )
    print(
        f"deltas are medians of {rounds} paired rounds against "
        f"'{reference}'; the (twin) row is the same configuration "
        f"built twice, so its delta is the null"
    )
    return rows


def pick_winner(rows: List[dict]) -> Tuple[str, int]:
    """Return the fastest (preconditioner, order) at minimum failures."""
    candidates = [
        row for row in rows if not row["config"].endswith("(twin)")
    ]
    min_failed = min(row["failed"] for row in candidates)
    viable = [
        row for row in candidates if row["failed"] == min_failed
    ]
    best = min(viable, key=lambda row: row["kernel_ms"])
    return best["preconditioner"], best["preconditioner_order"]


def part2_rows(algorithm: str, winner: Tuple[str, int]):
    """Return (label, correction, precond, order, variant) rows."""
    precond, order = winner
    tag = f"{precond}-{order}" if precond != "none" else "none"
    if algorithm == "rosenbrock23":
        # Linearly implicit: one inherently cached row per solver.
        return [
            (f"bicgstab {tag} cached", "bicgstab", precond, order,
             "cached"),
            (f"mr {tag} cached", "minimal_residual", precond, order,
             "cached"),
            ("lu cached", "lu", "none", 0, "cached"),
        ]
    rows = [
        (f"bicgstab {tag} exact", "bicgstab", precond, order,
         "exact"),
        (f"bicgstab {tag} inexact", "bicgstab", precond, order,
         "inexact"),
        (f"mr {tag} exact", "minimal_residual", precond, order,
         "exact"),
        (f"mr {tag} inexact", "minimal_residual", precond, order,
         "inexact"),
        ("lu exact", "lu", "none", 0, "exact"),
    ]
    if algorithm == "kvaerno3":
        # FIRK has no non-prefactored frozen direct solve.
        rows.append(("lu inexact", "lu", "none", 0, "inexact"))
    rows.append(("lu prefactored", "lu", "none", 0, "prefactored"))
    return rows


def append_csv(path: Path, system: str, algorithm: str, part: str,
               n_runs: int, rows: List[dict]) -> None:
    """Append one sweep's rows to the results CSV."""
    fields = [
        "system", "algorithm", "part", "n_runs", "config", "variant",
        "correction", "preconditioner", "preconditioner_order",
        "kernel_ms", "kernel_delta_pct", "wall_ms", "waves",
        "failed", "flags",
    ]
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    row,
                    system=system,
                    algorithm=algorithm,
                    part=part,
                    n_runs=n_runs,
                )
            )


def run_group(system_name: str, system, spec, algorithm: str,
              n_runs: int, args, csv_path: Path,
              setup_s: float) -> None:
    """Run both parts for one (system, algorithm) group."""
    duration = spec["duration"]
    grid_builder = spec["grid"]
    inits = params = None

    if args.part2_only is not None:
        winner = (args.part2_only[0], int(args.part2_only[1]))
        print(
            f"part 1 skipped; part 2 uses {winner[0]} order "
            f"{winner[1]}"
        )
    else:
        print()
        print(
            f"--- {system_name} / {algorithm}: part 1, bicgstab "
            f"preconditioners, {n_runs} runs ---"
        )
        entries = []
        part1 = part1_preconditioners(algorithm)
        part1.append(
            (f"{part1[0][0]} (twin)",) + tuple(part1[0][1:])
        )
        for label, precond, order in part1:
            solver = build_solver(
                system, algorithm, spec, "bicgstab", precond, order,
                "exact",
            )
            if inits is None:
                inits, params = grid_builder(solver, n_runs)
            entries.append(
                Config(
                    label, solver, "exact", "bicgstab", precond, order
                )
            )
        entries, timed_out = prepare(
            entries, inits, params, duration, n_runs, system_name,
            algorithm, args.timeout_factor, setup_s,
        )
        rows = run_sweep(
            f"{system_name} / {algorithm} / part 1", entries, inits,
            params, duration, args.rounds, args.block, args.min_count,
        )
        append_csv(
            csv_path, system_name, algorithm, "part1", n_runs,
            rows + [dropped_row(entry) for entry in timed_out],
        )
        winner = pick_winner(rows)
        print(
            f"fastest preconditioner: {winner[0]} order {winner[1]}"
        )
        del entries

    print()
    print(
        f"--- {system_name} / {algorithm}: part 2, solver x variant, "
        f"{n_runs} runs ---"
    )
    specs = part2_rows(algorithm, winner)
    specs.append((f"{specs[0][0]} (twin)",) + tuple(specs[0][1:]))
    entries = []
    for label, correction, precond, order, variant in specs:
        solver = build_solver(
            system, algorithm, spec, correction, precond, order,
            variant,
        )
        if inits is None:
            inits, params = grid_builder(solver, n_runs)
        entries.append(
            Config(label, solver, variant, correction, precond, order)
        )
    entries, timed_out = prepare(
        entries, inits, params, duration, n_runs, system_name,
        algorithm, args.timeout_factor, setup_s,
    )
    rows = run_sweep(
        f"{system_name} / {algorithm} / part 2", entries, inits,
        params, duration, args.rounds, args.block, args.min_count,
    )
    append_csv(
        csv_path, system_name, algorithm, "part2", n_runs,
        rows + [dropped_row(entry) for entry in timed_out],
    )
    del entries


def parse_args(argv: Optional[Sequence[str]] = None):
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SYSTEMS),
        default=tuple(SYSTEMS),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block", type=int, default=9)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="trajectory count (default: two full waves at the "
        "device occupancy ceiling)",
    )
    parser.add_argument(
        "--fabbri-runs",
        type=int,
        default=None,
        help="fabbri trajectory count override (default: --n-runs)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("linear_solver_grid_results.csv"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one round of three solves per config for a fast "
        "end-to-end check",
    )
    parser.add_argument(
        "--timeout-factor",
        type=float,
        default=3.0,
        help="drop a config when its trial solve exceeds this "
        "multiple of the reference warm-up (plus setup allowance)",
    )
    parser.add_argument(
        "--part2-only",
        nargs=2,
        metavar=("PRECOND", "ORDER"),
        default=None,
        help="skip part 1 and run part 2 with this preconditioner",
    )
    parser.add_argument(
        "--probe",
        nargs=6,
        metavar=(
            "SYSTEM", "ALGORITHM", "CORRECTION", "PRECOND", "ORDER",
            "VARIANT",
        ),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def run_probe(args) -> None:
    """Build one configuration and run a single solve."""
    name, algorithm, correction, precond, order, variant = args.probe
    spec = SYSTEMS[name]
    system = spec["build"]()
    solver = build_solver(
        system, algorithm, spec, correction, precond, int(order),
        variant,
    )
    inits, params = spec["grid"](solver, args.n_runs)
    solve_once(solver, inits, params, spec["duration"] / 100.0)
    print("SETUP_DONE", flush=True)
    solve_once(solver, inits, params, spec["duration"])
    print("PROBE_OK", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the requested grid."""
    if CUDA_SIMULATION:
        raise SystemExit(
            "linear_solver_grid.py measures kernel time on a real "
            "GPU; unset NUMBA_ENABLE_CUDASIM"
        )
    args = parse_args(argv)
    if args.probe is not None:
        run_probe(args)
        return
    if args.smoke:
        args.rounds = 1
        args.block = 3
        args.min_count = 1
    auto_runs = default_n_runs()
    n_runs = args.n_runs if args.n_runs is not None else auto_runs
    fabbri_runs = (
        args.fabbri_runs if args.fabbri_runs is not None else n_runs
    )
    device = cuda.get_current_device()
    name = device.name
    if isinstance(name, bytes):
        name = name.decode()
    print(
        f"device: {name} ({device.MULTIPROCESSOR_COUNT} SMs); "
        f"n_runs {n_runs} (auto {auto_runs}), fabbri {fabbri_runs}"
    )
    start = perf_counter()
    for system_name in args.systems:
        spec = SYSTEMS[system_name]
        build_start = perf_counter()
        system = spec["build"]()
        setup_s = 60.0 + (perf_counter() - build_start)
        system_runs = (
            fabbri_runs if system_name == "fabbri" else n_runs
        )
        for algorithm in args.algorithms:
            run_group(
                system_name, system, spec, algorithm, system_runs,
                args, args.csv, setup_s,
            )
    print()
    print(
        f"grid complete in {(perf_counter() - start) / 60.0:.1f} "
        f"minutes; results in {args.csv}"
    )


if __name__ == "__main__":
    main()
